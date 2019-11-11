import scipy.io 
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

#
# Import LIVE Image Data
#
import re
from os import listdir
from os.path import isfile, join

head = '../../LIVE/'
paths = [d+'/' for d in listdir(head) if d in ['jpeg','jp2k', 'refimgs']]
jpeg = []
jp2k = []
ref = []

for path in paths:
    for f in listdir(head+path):
        if re._compile('.*?bmp',flags=0).match(f):
            if f=='jpeg':
                jpeg.append(head+path+f)
            else if f=='jp2k':
                jp2k.append(head+path+f)
            else if f=='refimgs':
                ref.append(head+path+f)

jpeg = np.asarray(jpeg)
jp2k = np.asarray(jp2k)
ref = np.asarray(ref)

image_paths = np.append(jpeg, jp2k, ref)
X = [plt.imread(i) for i in image_paths]

# Get labels (scores)
from jpeg_scores import *
Y = np.append(np.asarray(jpeg_labels),
              np.asarray(jp2k_labels),
              np.zeros(ref.shape))


# Split data for testing
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)

#
# The Question:
# Can this learn what it means to be a high
# quality image? 
#

import cv2
import torch
from torchvision.transforms import *
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#params = {
#          'hl_depth': 5,
#          'num_hl': 2,
#          'dropout_rate': 0.0,
#          'batch_size': 10
#         }
#classifier = Sequential()
#
## First Hidden Layer
#classifier.add(Dense(params['hl_depth'], 
#                         activation='relu',
#                         kernel_initializer='random_normal',
#                         input_dim=n_features))
#classifier.add(Dropout(params['dropout_rate']))
#
## Hidden Layers
#for hl in range(params['num_hl']-1):
#    classifier.add(Dense(params['hl_depth'], 
#                         activation='relu',
#                         kernel_initializer='random_normal'))
#    classifier.add(Dropout(params['dropout_rate']))
#      
## Output Layer
#classifier.add(Dense(1, 
#                     activation='sigmoid',
#                     kernel_initializer='random_normal'))
#
#
#
## Compiling Network
## Adam stands for Adaptive moment estimation. Adam = RMSProp + Momentum
## Momentum - takes into account past gradients to smooth out gradient descent
#classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
#
## Fitting data
#classifier.fit(X_train, y_train, params['batch_size'], epochs=100)
#
## Evaluate
#eval_model = classifier.evaluate(X_train, y_train)
#y_pred = classifier.predict(X_test)
#y_pred_cm = (y_pred>0.5)
#
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_cm)
#
#
