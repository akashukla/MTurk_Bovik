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
# Generate Data
#
from lib.dataload import *
from lib.denoise import *
from lib.normalize import *
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

# Make the data
im_dict, data = make_predata(csv_files,column_headers)

# Denoise and get cleaned data
percentile = 90
cleaned_data = outlier_denoising(im_dict, data, percentile)
image_names, s1_scores, s2_scores = format_data(cleaned_data)

#
# Generate class directories
#
classes = np.arange(0,50).astype('str')

os.mkdir('sorted_data')
os.mkdir('sorted_data/train')
os.mkdir('sorted_data/val')
for class_i in classes:
     os.mkdir('sorted_data/train/'+class_i)
     os.mkdir('sorted_data/val/'+class_i)
     
for i in range(len(image_names)):
     dest_tr = 'sorted_data/' + '/train/'+ (s1_scores[i].round().astype('int')).astype('str')
     dest_va = 'sorted_data/' + '/val/' + (s1_scores[i].round().astype('int')).astype('str')
     os.system('cp 8k_data/%s %s'%(image_names[i], dest_tr))
     os.system('cp 8k_data/%s %s'%(image_names[i], dest_va))


#
# Creating the dataloaders
#
kon_path = 'sorted_data/'

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'val' : transforms.Compose([
        transforms.ToTensor()
    ])
}

image_set = {x: datasets.ImageFolder(os.path.join(kon_path,x),
                data_transforms[x])
                for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_set[x], 
                #batch_size=4, shuffle=True, num_workers=4) 
                batch_size=1, shuffle=True, num_workers=1) 
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_set[x]) for x in ['train', 'val']}
class_names = image_set['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



