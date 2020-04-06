import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Access golden image access methods
import pandas as pd
import numpy.random as npr
from numpy.random import permutation, shuffle
#From arkan's hw q1
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
#From arkan's hw q2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def load_batch_data():
    b1 = pd.read_csv('batch-data/batch1_results.csv')
    b2 = pd.read_csv('batch-data/batch2_results.csv')
    b3 = pd.read_csv('batch-data/batch3_results.csv')
    b1 = b1[b1.loc[:,'AssignmentStatus'] == 'Approved']
    b2 = b2[b2.loc[:,'AssignmentStatus'] == 'Approved']
    b3 = b3[b3.loc[:,'AssignmentStatus'] == 'Approved']
    b1 = b1[b1.loc[:,'Answer.set_number'] != 'initial']
    b2 = b2[b2.loc[:,'Answer.set_number'] != 'initial']
    b3 = b3[b3.loc[:,'Answer.set_number'] != 'initial']
    
    b1_setnum=b1.loc[:,'Answer.set_number']
    b2_setnum=b2.loc[:,'Answer.set_number']
    b3_setnum=b3.loc[:,'Answer.set_number']
    batch_setnums = np.append(np.append(b1_setnum,b2_setnum),b3_setnum).astype('int64')
    
    
    b1_svals=np.array([np.fromstring(np.array(b1.loc[:,'Answer.slider_values'])[i],dtype='int',sep=',') for i in range(b1.shape[0])])
    b2_svals=np.array([np.fromstring(np.array(b2.loc[:,'Answer.slider_values'])[i],dtype='int',sep=',') for i in range(b2.shape[0])])
    b3_svals=np.array([np.fromstring(np.array(b3.loc[:,'Answer.slider_values'])[i],dtype='int',sep=',') for i in range(b3.shape[0])])
    #batch_svals=np.append(np.append(b1_svals,b2_svals),b3_svals)
    batch_svals=np.row_stack((b1_svals,b2_svals,b3_svals))
   
    
    hdo=np.genfromtxt('hit_set_data/hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]
    sval_dict = {}
    svals_all = {}
    sval_count = {}
    sval_avg = {}
    for i in range(len(batch_setnums)):
        setnum = batch_setnums[i]
        set_svals = batch_svals[i]
        set_names = hdo[setnum-1]
        for j in range(len(set_names)):
            set_name = set_names[j]
            svals_all[set_name] = []
            sval_dict[set_name] = 0
            sval_count[set_name] = 0
            
    for i in range(len(batch_setnums)):
        setnum=batch_setnums[i]
        set_svals = batch_svals[i]
        set_names = hdo[setnum-1]
        for j in range(len(set_names)):
            set_name=set_names[j]
            svals_all[set_name].append(set_svals[j])
            sval_dict[set_name]+=set_svals[j]
            sval_count[set_name]+=1
    
    for i in range(len(batch_setnums)):
        setnum=batch_setnums[i]
        set_svals = batch_svals[i]
        set_names = hdo[setnum-1]
        for j in range(len(set_names)):
            set_name = set_names[j]
            sval_avg[set_name] = sval_dict[set_name]/sval_count[set_name]
            
    sval_avg.pop('AVA__330138.jpg')
    svals_all.pop('AVA__330138.jpg')
    #return svals_all, sval_avg
    image_scores=[]
    image_names=[]
    for key, val in svals_all.items():
            temp = [key,val]
            image_names.append(key)
            image_scores.append(val)

    image_avgs=[]
    for key, val in sval_avg.items():
            temp = [key,val]
            image_avgs.append(val)
    return image_names,image_scores,image_avgs,svals_all,sval_avg

        
                 
# x = np.zeros((len(sval_avg),375, 500, 3),dtype='int8')
# #x_path = np.zeros((len(sval_avg)),dtype='str')
# x_path=[]
# y = np.zeros(len(sval_avg),dtype='float64')
# i=0
# for sv in sval_avg:
#     x_path.append('8k_data/'+sv)
#     x[i]=plt.imread(('8k_data/'+sv))
#     y[i]=sval_avg[sv]
#     i+=1
# x_path=np.array(x_path)
# np.save('x_path',x_path)
# np.save('x',x)
# np.save('y',y)
