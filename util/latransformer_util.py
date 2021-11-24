from __future__ import print_function

import os
import time
import random
import zipfile
from itertools import chain

import timm
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from LATransformer.model import ClassBlock, LATransformer
from LATransformer.utils import save_network, update_summary


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LaTransformerUtil(object):
    def __init__(self, device=None) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # training parameters
        # self.batch_size = 32
        # self.num_epochs = 30
        # self.lr = 3e-4
        # self.gamma = 0.7
        # self.unfreeze_after = 2
        # self.lr_decay = 0.8
        # self.lmbd = 8

        self._init_data_load()

    def _init_data_load(self):
        transform_train_list = [
            transforms.Resize((224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        transform_val_list = [
            transforms.Resize(size=(224, 224), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.data_transforms = {
            "train": transforms.Compose(transform_train_list),
            "val": transforms.Compose(transform_val_list),
        }

    def load_dateset(self):
        image_datasets = {}
        data_dir = "data/Market-Pytorch/Market/"

        image_datasets["train"] = datasets.ImageFolder(
            os.path.join(data_dir, "train"), self.data_transforms["train"]
        )
        image_datasets["val"] = datasets.ImageFolder(
            os.path.join(data_dir, "val"), self.data_transforms["val"]
        )
        train_loader = DataLoader(
            dataset=image_datasets["train"], batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            dataset=image_datasets["val"], batch_size=self.batch_size, shuffle=True
        )

        self.class_names = image_datasets["train"].classes

        return (self.class_names, train_loader, valid_loader)

    def load_vit_model(self):
        # Load pre-trained ViT
        vit_base = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=751
        )
        vit_base = vit_base.to(self.device)
        vit_base.eval()
        return vit_base

    def validate(self, model, loader, loss_fn, epoch, print_message=True):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        model.eval()
        epoch_accuracy = 0
        epoch_loss = 0
        end = time.time()
        last_idx = len(loader) - 1

        running_loss = 0.0
        running_corrects = 0.0

        with torch.no_grad():
            for input, target in tqdm(loader):

                input, target = input.to(self.device), target.to(self.device)

                output = model(input)

                score = 0.0
                sm = nn.Softmax(dim=1)
                for k, v in output.items():
                    score += sm(output[k])
                _, preds = torch.max(score.data, 1)

                loss = 0.0
                for k, v in output.items():
                    loss += loss_fn(output[k], target)

                batch_time_m.update(time.time() - end)
                acc = (preds == target.data).float().mean()
                epoch_loss += loss / len(loader)
                epoch_accuracy += acc / len(loader)

                if print_message:
                    print(
                        f"Epoch : {epoch+1} - val_loss : {epoch_loss:.4f} - val_acc: {epoch_accuracy:.4f}",
                        end="\r",
                    )

        metrics = OrderedDict(
            [
                ("val_loss", epoch_loss.data.item()),
                ("val_accuracy", epoch_accuracy.data.item()),
            ]
        )
        return metrics

    def train_one_epoch(
        self,
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        print_message=True,
    ):
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()

        model.train()
        epoch_accuracy = 0
        epoch_loss = 0
        end = time.time()

        for data, target in tqdm(loader):
            data, target = data.to(self.device), target.to(self.device)

            data_time_m.update(time.time() - end)

            optimizer.zero_grad()
            output = model(data)
            score = 0.0
            sm = nn.Softmax(dim=1)
            for k, v in output.items():
                score += sm(output[k])
            _, preds = torch.max(score.data, 1)

            loss = 0.0
            for k, v in output.items():
                loss += loss_fn(output[k], target)
            loss.backward()

            optimizer.step()

            batch_time_m.update(time.time() - end)

            acc = (preds == target.data).float().mean()

            epoch_loss += loss / len(loader)
            epoch_accuracy += acc / len(loader)

            if print_message:
                print(
                    f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}",
                    end="\r",
                )

        return OrderedDict(
            [
                ("train_loss", epoch_loss.data.item()),
                ("train_accuracy", epoch_accuracy.data.item()),
            ]
        )

    def freeze_all_blocks(self, model):
        frozen_blocks = 12
        for block in model.model.blocks[:frozen_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_blocks(self, model, amount=1):
        for block in model.model.blocks[11 - amount :]:
            for param in block.parameters():
                param.requires_grad = True
        return model

    def Train(
        self,
        num_epochs=30,
        batch_size=32,
        lr=3e-4,
        gamma=0.7,
        unfreeze_after=2,
        lr_decay=0.8,
        lmbd=8,
        print_message=True,
    ):

        # ## Training Loop
        # Create LA Transformer
        vit_base = self.load_vit_model()
        model = LATransformer(vit_base, lmbd).to(self.device)
        if print_message:
            print(model.eval())

        # loss function
        criterion = nn.CrossEntropyLoss()

        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=lr)

        # scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        self.freeze_all_blocks(model)

        best_acc = 0.0
        y_loss = {}  # loss history
        y_loss["train"] = []
        y_loss["val"] = []
        y_err = {}
        y_err["train"] = []
        y_err["val"] = []
        print("training...")
        output_dir = ""
        best_acc = 0
        name = f"la_with_lmbd_{lmbd}"

        try:
            os.mkdir("models/" + name)
        except:
            pass
        output_dir = "models/" + name
        unfrozen_blocks = 0

        (class_names, train_loader, valid_loader) = self.load_dateset()

        for epoch in range(num_epochs):

            if epoch % unfreeze_after == 0:
                unfrozen_blocks += 1
                model = self.unfreeze_blocks(model, unfrozen_blocks)
                optimizer.param_groups[0]["lr"] *= lr_decay
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                if print_message:
                    print(
                        "Unfrozen Blocks: {}, Current lr: {}, Trainable Params: {}".format(
                            unfrozen_blocks,
                            optimizer.param_groups[0]["lr"],
                            trainable_params,
                        )
                    )

            train_metrics = self.train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                criterion,
                lr_scheduler=None,
                saver=None,
                print_message=print_message,
            )

            eval_metrics = self.validate(
                model, valid_loader, criterion, epoch=epoch, print_message=print_message
            )

            # update summary
            update_summary(
                epoch,
                train_metrics,
                eval_metrics,
                os.path.join(output_dir, "summary.csv"),
                write_header=True,
            )

            # deep copy the model
            last_model_wts = model.state_dict()
            if eval_metrics["val_accuracy"] > best_acc:
                best_acc = eval_metrics["val_accuracy"]
                save_network(model, epoch, name)
                print("SAVED!")

        print("training finished.")
