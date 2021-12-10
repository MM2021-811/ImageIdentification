from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from typing import Optional, List, Callable
from torchvision.models.googlenet import BasicConv2d
import time
import numpy as np
import torchvision.models as models
from util.trainingutil import (
    AlphaAlexNet,
    SiameseAlexNet,
    ParameterError,
    AlphaBgTransform,
    SiameseLoader,
)
import os
import torchsummary
from pprint import pprint

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data1, data2, labels) in enumerate(train_loader):
        data1, data2, labels = (
            torch.tensor(data1,dtype=torch.float32).to(device),
            torch.tensor(data2,dtype=torch.float32).to(device),
            torch.tensor(labels,dtype=torch.float32).to(device),
        )
        optimizer.zero_grad()
        output = model(data1,data2)
        # loss = F.nll_loss(output, labels) # not supported
        loss = F.mse_loss(output,labels)
        # loss = F.cross_entropy(output, labels)
        # loss = F.l1_loss(output,labels)

        # pprint(output)
        # pprint(labels)
        # loss = torch.square(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}".format(
                    epoch,
                    batch_idx * len(data1),
                    len(train_loader),
                    100.0 * batch_idx * len(data1) / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data1, data2, labels) in enumerate(test_loader):
            data1, data2, labels = (
                torch.tensor(data1).to(device),
                torch.tensor(data2).to(device),
                torch.tensor(labels).to(device),
            )

            output = model(data1,data2)
            # sum up batch loss
            test_loss +=  F.l1_loss(output,labels).item()
            # get the index of the max log-probability
            output = torch.round(output)
            y = output[output == labels]

            correct += torch.sum(y)

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / len(test_loader)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.12f}%)\n".format(
            test_loss,
            correct,
            len(test_loader),
            test_accuracy,
        )
    )
    return test_accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485], std=[0.229]),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    train_loader = SiameseLoader(batch_size=60)
    test_loader = SiameseLoader(batch_size=250,train=False)

    model = SiameseAlexNet(device=device).to(device)
    model.train()
    
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    start = time.time()
    # load model from internal training
    checkpoint_model = "./models/bottle_siamese_tmp.pth"
    mode_saved_file = "./models/bottle_siamese.pth"

    if os.path.exists(checkpoint_model):
        model.load_state_dict(torch.load(checkpoint_model))
    
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)        
        scheduler.step()

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_model)
            if args.save_model:
                torch.save(model.state_dict(), mode_saved_file)
        
    end = time.time()
    print(f"Elapsed Time: {end - start}")


if __name__ == "__main__":
    main()
