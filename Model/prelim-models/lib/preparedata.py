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
# Note this part may take some time
from lib.dataload import *
from lib.denoise import *
from lib.normalize import *

csv_files = ['lib/batch-data/batch1_results.csv',
             'lib/batch-data/batch2_results.csv',
             'lib/batch-data/batch3_results.csv',
             'lib/batch-data/batch4_results.csv',
             'lib/batch-data/batch5_results.csv',
             'lib/batch-data/batch6_results.csv']

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




