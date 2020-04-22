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
import scipy
import scipy.stats as ss

# PyTorch libraries and modules
import torch
from torch.utils import data
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

import multiprocessing as mp

#
# GENERATE DATA
#
from lib.evalmodel import *
# Prepare all the data
from lib.preparedata import *

#
# Set up data format for regression based tl
#
tensor_x = torch.Tensor(X1_train)
tensor_y = torch.Tensor(y1_train)

dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset, num_workers=mp.cpu_count(), batch_size=64, shuffle=True)

# FOR MODEL 2
#model2 = models.resnet18(pretrained=True)
#for param in model2.parameters():
#    param.requires_grad = False
#
#num_ftrs = model2.fc.in_features
#net2 = nn.Sequential(nn.Linear(num_ftrs, 100), 
#                    nn.ReLU(),
#                    nn.Linear(100,50), 
#                    nn.ReLU(),
#                    nn.Linear(50,10), 
#                    nn.ReLU(),
#                    nn.Linear(10,1))
#net2.apply(init_weights)
#model2.fc = net2
#model2 = model2.to(device)
#
import time
import copy
def train_model(model, criterion, optimizer, scheduler, device, num_epochs=25):
    print('In train_model')
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
        # End epoch iterations

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# ERROR EVALUATIONS
def custom_err(y, yhat, thresh=5):
    diffs = y.reshape(y.shape[0]) - yhat.reshape(yhat.shape[0])
    diffs = np.abs(diffs)
    err = float(float(np.count_nonzero(diffs > thresh))/float(len(diffs)))
    return err


def srcc_err(y, yhat):
    y,yhat = y.reshape(y.shape[0]), yhat.reshape(yhat.shape[0])
    corr = ss.spearmanr(y, yhat)
    return corr

# ################################################### #
#                                                     #
#                 General Training                    #
#                                                     #
# ################################################### #

from torch.optim import lr_scheduler

def train_with_param(params):
    print('In train_with_param')

    # MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0001)
    
    best_model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, num_epochs=20)
    
    y_train, yhat_train = evaluate_model(X1_train, y1_train, model=best_model, device=device) 
    y_test, yhat_test = evaluate_model(X1_test, y1_test, model=best_model, device=device)
   
    # Find errors
    train_error = custom_err(y_train, yhat_train, thresh=5)
    test_error = custom_err(y_test, yhat_test, thresh=5)
    train_spr = srcc_err(y_train, yhat_train)
    test_spr = srcc_err(y_test, yhat_test)
 
    # Uncomment if you wnat to use custom evaluation
    # return train_error, test_error, train_spr, test_spr
    return y_test, yhat_test, y_train, yhat_train


# ############################# #
#                               #
# Specify your hyperparams here #
#                               #
# ############################# #

# Tell Satan not to spawn children
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

# Cross Validation
if __name__ == '__main__':
    # cores = mp.cpu_count() # Generally works if ltos of memory available
    # cores = 8 # Uncoment this line and reassign if memory runs out
    start = -5
    stop = 0
    step = (stop-start)/cores
    print('lrs', np.r_[10.0]**np.r_[start:stop:step])

    cv_range = np.r_[10.0]**np.r_[start:stop:step]
    
    p = MyPool(cores)
    scores = p.map(train_with_param, cv_range)
    print('scores', scores)
    scores = np.array(scores)
    np.save('gen/learning_rates.npy', cv_range)
    np.save('gen/scores.npy', scores)
    p.close()
    p.join()
    print('all done')
    
