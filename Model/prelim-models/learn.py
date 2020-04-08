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
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

#
# GENERATE DATA
#
from dataload import *
from denoise import *
        
csv_files = ['batch-data/batch1_results.csv','batch-data/batch2_results.csv','batch-data/batch3_results.csv']
column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

im_dict, data = make_predata(csv_files,column_headers)

# Denoise and get cleaned data
# set denoise hyperparam
percentile = 90
cleaned_data = outlier_denoising(im_dict, data, percentile)
X, y1, y2 = format_data(cleaned_data)



