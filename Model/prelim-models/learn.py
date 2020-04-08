#
# LIB IMPORT
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

# torchvision for pre-trained models
from torchvision import datasets, transforms, models



#
# GENERATE DATA
#
from dataload import *
from denoise import *
from normalize import *

csv_files = ['batch-data/batch1_results.csv','batch-data/batch2_results.csv','batch-data/batch3_results.csv']
column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

im_dict, data = make_predata(csv_files,column_headers)

# Denoise and get cleaned data
percentile = 90
cleaned_data = outlier_denoising(im_dict, data, percentile)
X, y1, y2 = format_data(cleaned_data)
y1 = y1.astype(int)
y2 = y2.astype(int)

# Normalize via zero-centering images
#X = zero_center(X)

# Split into train and test
# NOTE: only trying slider 1 for now
test_size = 0.2
train_X, test_X, train_y, test_y = train_test_split(X, y1, test_size=test_size, random_state=13)

# Convert input images to torch format
train_X = train_X.reshape(train_X.shape[0], 3, 224, 224)
train_X = torch.from_numpy(train_X)

test_X = test_X.reshape(test_X.shape[0], 3, 224, 224)
test_X = torch.from_numpy(test_X)

# Convert output into torch format
train_y = train_y.astype(int)
train_Y = torch.from_numpy(train_y)

test_y = test_y.astype(int)
test_y = torch.from_numpy(test_y)


#
# PRETRAINED MODEL
#

# Load model
model = models.vgg16_bn(pretrained=True)

# Freeze the model
for param in model.parameters():
    param.requires_grad = False

# Set device to gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Append a regressor/classifier
model = models.densenet121(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.require_grad = False

# Create custom model to append 
fc = nn.Sequential(
    nn.Linear(1024, 460),
    nn.ReLU(),
    nn.Dropout(0.4),
    
    nn.Linear(460,50),
    nn.LogSoftmax(dim=1)
    
)

model.classifier = fc
criterion = nn.NLLLoss()

# Choose optimizer
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device)

epochs = 1
valid_loss_min = np.Inf

import time
for epoch in range(epochs):
    start = time.time()

    model.train()

    train_loss = 0.0
    valid_loss = 0.0

    for inputs, labels in zip(train_X, train_y):
        # Send to available device
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(0)
        inputs, labels = Variable(torch.from_numpy(np.asarray(inputs))), Variable(torch.from_numpy(np.asarray(labels)))
        optimizer.zero_grad()
        
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        model.eval()

    with torch.no_grad():
        accuracy = 0
        
        for inputs, labels in zip(test_X, test_y):
            #inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(0)
            inputs, labels = Variable(torch.from_numpy(np.asarray(inputs))), Variable(torch.from_numpy(np.asarray(labels)))
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


    train_loss = train_loss/len(train_X)
    valid_loss = valid_loss/len(test_X)
    valid_accuracy = accuracy/len(test_X) 
    
    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(epoch + 1, train_loss, valid_loss, valid_accuracy))


    print(f"Time per epoch: {(time.time() - start):.3f} seconds")


