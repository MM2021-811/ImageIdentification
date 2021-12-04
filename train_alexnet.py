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
import torchvision.models as models
from util.trainingutil import AlphaAlexNet, ParameterError, AlphaBgTransform
import os
import torchsummary 
from pprint import pprint

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        #loss = F.mse_loss(output,target)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    # model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # n_output = target.shape[0]
            # output = torch.zeros((n_output,model.num_classes)).to(device)
            # for i in range(n_output):
            #     o1 = model(data)
            #     output[i] = o1

            output = model(data)
            # sum up batch loss
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            #test_loss += F.mse_loss(output,target,reduce="sum").item()
            test_loss += F.cross_entropy(output,target,reduce="sum").item()
            # get the index of the max log-probability
            # todo: debug error

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       "multiprocessing_context":'spawn',
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    #GoogleLeNet requirements
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1]
    # and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transform = transform = transforms.Compose([AlphaBgTransform()])
    dataset1 = datasets.CIFAR100('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.CIFAR100('../data', train=False,
                              transform=transform)

    # dataset1 = datasets.ImageFolder('../data/ILSVRC/Data/DET/train',transform=transform)
    # dataset2 = datasets.ImageFolder('../data/ILSVRC/Data/DET/test',transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = Net(init_weights=True).to(device)
    # model = models.vgg16(pretrained=False,num_classes=100).to(device)
    model = models.alexnet(pretrained=False,num_classes=100).to(device)
    # model = AlphaAlexNet(num_classes=100).to(device)
    model.train()

    pprint(torchsummary.summary(model,input_size=(3,224,224),batch_size=64))
    
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    start = time.time()
    # load model from internal training
    checkpoint_model = f'./models/CIFAR100_AlexNet_tmp.pt' 
    if os.path.exists(checkpoint_model):
        model.load_state_dict(torch.load(checkpoint_model))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        torch.save(model.state_dict(), checkpoint_model)
    end = time.time()
    print(f"Elapsed Time: {end - start}")

    if args.save_model:
        torch.save(model.state_dict(), "CIFAR100_AlexNet.pt")


if __name__ == '__main__':
    main()
