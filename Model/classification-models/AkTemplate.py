#
# Import all needed libraries
#

#
# LIB IMPORTS
#
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
#import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

# torchvision for pre-trained models
from torchvision import datasets, transforms, models



#
# GENERATE DATA
#
# Note this part may take some time the first time you run this
# All sequential runs will be faster
#from lib.preparedata import *
from lib.evalmodel import *

#
# Set up data format for regression based tl
#

import torch
from torch.utils import data
X1= np.load('image_array.npy')
X1=np.reshape(X1, (X1.shape[0],3,375,500))
y1= np.load('s1.npy')
X1_train = X1[0:4*len(X1)//5]
y1_train = y1[0:4*len(y1)//5]
X1_test = X1[4*len(X1)//5:]
y1_test = y1[4*len(y1)//5:]

tensor_x = torch.Tensor(X1_train)
tensor_y = torch.Tensor(y1_train)

import multiprocessing as mp
dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset, num_workers=mp.cpu_count(), batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# MODEL 1
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


num_ftrs = model.fc.in_features

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

net = nn.Sequential(nn.Linear(num_ftrs,50), nn.Linear(50, 1))
net.apply(init_weights)
model.fc = net
model = model.to(device)


# FOR MODEL 2
model2 = models.resnet18(pretrained=True)
for param in model2.parameters():
    param.requires_grad = False

num_ftrs = model2.fc.in_features
net2 = nn.Sequential(nn.Linear(num_ftrs, 100), 
                    nn.ReLU(),
                    nn.Linear(100,50), 
                    nn.ReLU(),
                    nn.Linear(50,10), 
                    nn.ReLU(),
                    nn.Linear(10,1))
net2.apply(init_weights)
model2.fc = net2
model2 = model2.to(device)





#
# General Training
#
criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Decay LR by a factor of 0.1 every 7 epochs
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0001)




import time
import copy
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #class_dict=(dataloaders['train'].dataset.class_to_idx)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('LR: ' , scheduler.get_lr())
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(outputs)
                    print(labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
                running_corrects += torch.sum(abs(preds - labels.data) > 5)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



#model = train_model(model, criterion, optimizer, exp_lr_scheduler,
#                       num_epochs=20)
#torch.save(model, 'newarch1')

model2 = train_model(model2, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
torch.save(model2, 'newarch2')


#y, yhat = evaluate_model(X1_test, y1_test, model=model)

t_x = torch.Tensor(X1_test) 
t_y = torch.Tensor(y1_test) 
 
d = data.TensorDataset(t_x, t_y) 
dataloader_test = data.DataLoader(d, num_workers=mp.cpu_count())

y = np.zeros(len(y1_test)) 
yhat = np.zeros(len(y1_test)) 
i=0 
for inputs, labels  in dataloader_test: 
    inputs = inputs.to(device) 
    labels = labels.to(device) 
    outputs = model(inputs) 
    y[i] = labels[0].data.cpu().detach().numpy() 
    yhat[i] = outputs[0][0].data.cpu().detach().numpy() 
    i+=1 

