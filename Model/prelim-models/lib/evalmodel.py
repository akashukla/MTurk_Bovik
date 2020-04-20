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


def evaluate_model(X_test, y_test, model):
    t_x = torch.Tensor(X_test)
    t_y = torch.Tensor(y1_test)
    
    d = data.TensorDataset(t_x, t_y)
    dataloader_test = data.DataLoader(d, num_workers=mp.cpu_count())
    
    for inputs, labels  in dataloader_test:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
 
    return labels, outputs
