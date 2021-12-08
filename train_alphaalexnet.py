# source: https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
# get_ipython().run_line_magic('matplotlib', 'inline')
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
from util.trainingutil import AlphaAlexNet
from tqdm import tqdm
import os

# %%
#creating a dinstinct transform class for the train, validation and test dataset
tranform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#preparing the train, validation and test dataset
torch.manual_seed(43)
train_ds = CIFAR10("../data/", train=True, download=True, transform=tranform_train) #40,000 original images + transforms
val_size = 10000 #there are 10,000 test images and since there are no transforms performed on the test, we keep the validation as 10,000
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size]) #Extracting the 10,000 validation images from the train set
test_ds = CIFAR10("../data/", train=False, download=True, transform=tranform_test) #10,000 images

#passing the train, val and test datasets to the dataloader
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda

model = AlphaAlexNet() #to compile the model
model = model.to(device=device) #to send the model for training on either cuda or cpu

model_file = "./models/alexnet_alpha.pth"
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file)) 
    model.to(device)

## Loss and optimizer
# learning_rate = 1e-4 #I picked this because it seems to be the most used by experts
learning_rate = 1e-7 #I picked this because it seems to be the most used by experts
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

best_accuracy = 0
for epoch in range(500): #I decided to train the model for 50 epochs
    loss_ep = 0
    
    for batch_idx, (data, targets) in tqdm(enumerate(train_dl)):
        data = data.to(device=device)
        targets = targets.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
        # print(f"Loss in epoch {epoch} Batch:{batch_idx} {loss_ep/len(train_dl)}")
    print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data,targets) in enumerate(val_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
                
        accuracy = float(num_correct) / float(num_samples) * 100
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_file)
        print(
            f"{epoch}: Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}"
        )    
    
    if best_accuracy > 92:
        print("Reached 82% Quit")
        break
                
        



# %%
