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
# GENERATE DATA IF NOT DONE ALREADY
#
# Note this part may take some time

import os.path
from lib.dataload import *
from lib.denoise import *
from lib.normalize import *

if os.path.isfile(os.getcwd()+'/X1_train.npy'):
    print('Fetching data ...')
    X1_train = np.load('X1_train.npy')
    X2_train = np.load('X2_train.npy')
    X1_test = np.load('X1_test.npy')
    X2_test = np.load('X2_test.npy')
    y1_train = np.load('y1_train.npy')
    y2_train = np.load('y2_train.npy')
    y1_test = np.load('y1_test.npy')
    y2_test = np.load('y2_test.npy')

else:
    csv_files = ['lib/batch-data/batch1_results.csv',
             'lib/batch-data/batch2_results.csv',
             'lib/batch-data/batch3_results.csv',
             'lib/batch-data/batch4_results.csv',
             'lib/batch-data/batch5_results.csv',
             'lib/batch-data/batch6_results.csv',
             'lib/batch-data/batch7_results.csv',
             'lib/batch-data/batch8_results.csv',
             'lib/batch-data/batch9_results.csv',
             'lib/batch-data/batch10_results.csv',
             'lib/batch-data/batch12_results.csv',
             'lib/batch-data/batch13_results.csv',
             'lib/batch-data/batch14_results.csv',
             'lib/batch-data/batch15_results.csv',
             'lib/batch-data/batch16_results.csv',
             'lib/batch-data/batch17_results.csv',
             'lib/batch-data/batch18_results.csv',
             'lib/batch-data/batch19_results.csv',
             'lib/batch-data/batch20_results.csv',
             'lib/batch-data/batch21_results.csv',
             'lib/batch-data/batch22_results.csv',
             'lib/batch-data/batch23_results.csv',
             'lib/batch-data/batch24_results.csv',
             'lib/batch-data/batch25_results.csv',
             'lib/batch-data/batch26_results.csv']
   
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
    X1_train, X1_test, y1_train, y1_test = train_test_split(Images, 
                                                          y1, 
                                                          test_size=test_size, 
                                                          random_state=123)
    X2_train, X2_test, y2_train, y2_test = train_test_split(Images, 
                                                          y2, 
                                                          test_size=test_size, 
                                                          random_state=123)
    
    
    np.save('X1_train', X1_train)
    np.save('X2_train', X2_train)
    np.save('X1_test', X1_test)
    np.save('X2_test', X2_test)
    np.save('y1_train', y1_train)
    np.save('y2_train', y2_train)
    np.save('y1_test', y1_test)
    np.save('y2_test', y2_test)
    
    print("Done preparing data")
    



