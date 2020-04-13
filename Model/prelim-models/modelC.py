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

# fastai
import fastai
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.imports import *



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
# X contains the image names
X, y1, y2 = format_data(cleaned_data)


# Split the data into test and train set
test_size = 0.2
X_train, X_test, y1_train, y1_test = train_test_split(X, 
                                                      y1, 
                                                      test_size=test_size, 
                                                      random_state=123)
X_train, X_test, y2_train, y2_test = train_test_split(X, 
                                                      y2, 
                                                      test_size=test_size, 
                                                      random_state=123)


# NOTE: Beyond this point we are only trying slider 1

#
# Set up data format for regression based tl
#
# FOR TRAINING DATA
df_train = pd.DataFrame(np.array([X_train, y1_train]).T, columns=['names', 'scores'])
df_train.scores = df_train.scores.astype('float')

img_path = '../data/8k_data/'
data_train = (ImageList
 .from_df(path=img_path,df=df_train)
 .split_by_rand_pct()
 .label_from_df(cols=1,label_cls=FloatList)
 .transform(get_transforms(), size=224)
 .databunch(bs=16)
 .normalize(imagenet_stats))

# FOR TESTING DATA
df_test = pd.DataFrame(np.array([X_test, y1_test]).T, columns=['names', 'scores'])
df_test.scores = df_test.scores.astype('float')

data_test = (ImageList
 .from_df(path=img_path,df=df_test)
 .split_by_rand_pct()
 .label_from_df(cols=1,label_cls=FloatList)
 .transform(get_transforms(), size=224)
 .databunch(bs=16)
 .normalize(imagenet_stats))


learn = cnn_learner(data_train, models.resnet50, metrics=[accuracy], true_wd=False)
learn.fit_one_cycle(5)
learn.save('base_test')

