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
import matplotlib.pyplot as plt

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

# fastai
import fastai
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.imports import *




#
# GENERATE DATA
#
# Note this part may take some time
from dataload import *
from denoise import *
from normalize import *

csv_files = ['batch-data/batch1_results.csv',
             'batch-data/batch2_results.csv',
             'batch-data/batch3_results.csv',
             'batch-data/batch4_results.csv',
             'batch-data/batch5_results.csv',
             'batch-data/batch6_results.csv']

column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

im_dict, data = make_predata(csv_files,column_headers)

# Denoise and get cleaned data
percentile = 90
cleaned_data = outlier_denoising(im_dict, data, percentile)

# X contains the image names
Images, X, y1, y2 = format_data(cleaned_data)

Images = Images.reshape(Images.shape[0], Images.shape[-1], Images.shape[1], Images.shape[2])

# Split the data into test and train set
test_size = 0.2
X_train, X_test, y1_train, y1_test = train_test_split(Images, 
                                                      y1, 
                                                      test_size=test_size, 
                                                      random_state=123)
X_train, X_test, y2_train, y2_test = train_test_split(Images, 
                                                      y2, 
                                                      test_size=test_size, 
                                                      random_state=123)


print("Done preparing data")



# NOTE: Beyond this point we are only trying slider 1

#
# Set up data format for regression based tl
#

import torch
from torch.utils import data

tensor_x = torch.Tensor(X_train)
tensor_y = torch.Tensor(y1_train)

dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = data.DataLoader(dataset)

#class Dataset(data.Dataset):
#  'Characterizes a dataset for PyTorch'
#  def __init__(self, list_IDs, labels):
#        'Initialization'
#        self.labels = labels
#        self.list_IDs = list_IDs
#
#  def __len__(self):
#        'Denotes the total number of samples'
#        return len(self.list_IDs)
#
#  def __getitem__(self, index):
#        'Generates one sample of data'
#        # Select sample
#        ID = self.list_IDs[index]
#
#        # Load data and get label
#        X = torch.load('../data/8k_data/' + ID)
#        y = self.labels[ID]
#
#        return X, y
    
    
# CUDA for PyTorch
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#if use_cuda:
#    cudnn.benchmark = True
#
# Parameters
#params = {'batch_size': 16,
#          'shuffle': True}
#
#training_set = Dataset(X_train, y1_train)
#training_generator = data.DataLoader(training_set, **params)
#
#validation_set = Dataset(X_test, y1_test)
#validation_generator = data.DataLoader(validation_set, **params)



#classes=np.arange(0,51).astype('str')
#kon_path = '../data/8k_data'
#
#data_transforms = {
#    'train': transforms.Compose([
#        transforms.ToTensor()
#    ]),
#    'val' : transforms.Compose([
#        transforms.ToTensor()
#    ])
#}
#
#image_set = {x: datasets.ImageFolder(os.path.join(kon_path,x),
#                data_transforms[x])
#                for x in ['train', 'val']}
#
#dataloaders = {x: torch.utils.data.DataLoader(image_set[x], 
#                #batch_size=4, shuffle=True, num_workers=4) 
#                batch_size=1, shuffle=True, num_workers=1) 
#                for x in ['train', 'val']}
#
#dataset_sizes = {x: len(image_set[x]) for x in ['train', 'val']}
#class_names = image_set['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Decay LR by a factor of 0.1 every 7 epochs
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




import time
import copy
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #class_dict=(dataloaders['train'].dataset.class_to_idx)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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



model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=2)






