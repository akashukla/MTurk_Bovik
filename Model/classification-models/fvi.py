#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
# Std lib imports
#
import os, sys, glob
import pandas as pd
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

#
# Our lib imports
#
from lib.dataload import *
from lib.features import *
from lib.denoise import *


# ### Generating a Scores Matrix

# In[2]:


# 
# Generate a dictionary of images with workers and scores
# to make into score matrix
#

# Specify files and relevant columns
csv_files = glob.glob('lib/batch-data/batch*')
column_headers = ['AssignmentStatus',
                  'Answer.set_number',
                  'WorkerId',
                  'Answer.slider_values',
                  'Answer.slider_values2']

dataDF = pd.DataFrame()

for f in csv_files:
    df = pd.read_csv(f)
    df = df[df.loc[:,'AssignmentStatus'] == 'Approved']
    df = df[df.loc[:,'Answer.set_number'] != 'initial']
    df = df.loc[:,column_headers]
    
    # Determine hit data file
    if re.match(r'.*batch[1-3]_',f) is not None:
        df['HitDataFile'] = np.repeat(0,len(df))
    elif re.match(r'.*batch[4-6]_',f) is not None:
        df['HitDataFile'] = np.repeat(1,len(df))
    else:
        df['HitDataFile'] = np.repeat(2,len(df))

    dataDF = pd.concat([dataDF,df], axis='index', ignore_index=True)

# Get all HIT Data files
hit_data_1_3 = np.genfromtxt('lib/batch-data/hit_data_1_3.csv',
                              delimiter=',',
                              dtype='str')[:,1:56]
hit_data_4_6 = np.genfromtxt('lib/batch-data/hit_data_4_6.csv',
                              delimiter=',',
                              dtype='str')[:,1:56]
hit_data_7 = np.genfromtxt('lib/batch-data/hit_data_7.csv',
                              delimiter=',',
                              dtype='str')[:,1:56]
# Get image names
im_names = pd.read_csv('../data/8k_image_names.csv',
                       header=None,
                       usecols=[0])
im_names = (im_names.loc[:,0]).tolist()
im_dict = {keys: None for keys in im_names}

for w in tqdm.tqdm(range(len(dataDF))):
    # Get worker stats
    worker_id = dataDF.loc[w,'WorkerId']
    set_num = int(dataDF.loc[w,'Answer.set_number'])
    slider1_vals = np.asarray(dataDF.loc[w,'Answer.slider_values'].split(','))[:-1].astype('int64')
    slider2_vals = np.asarray(dataDF.loc[w,'Answer.slider_values2'].split(','))[:-1].astype('int64')
    
    # Get set images by hit file
    if int(dataDF.loc[w,'HitDataFile']) == 0:
        set_names = hit_data_1_3[(set_num)%len(hit_data_1_3)]
    elif int(dataDF.loc[w,'HitDataFile']) == 1:
        set_names = hit_data_4_6[(set_num)%len(hit_data_4_6)]
    elif int(dataDF.loc[w,'HitDataFile']) == 2:
        set_names = hit_data_7[(set_num)%len(hit_data_7)]
    else:
        assert False, 'HIT Data File not assigned'
        
    # Populate im dict
    for n in range(len(set_names)):
        if im_dict[set_names[n]] is None:
            im_dict[set_names[n]] = np.asarray([worker_id,
                                                slider1_vals[n],
                                                slider2_vals[n]])
        else:
            im_dict[set_names[n]] = np.vstack((im_dict[set_names[n]],
                                               [worker_id,
                                                slider1_vals[n],
                                                slider2_vals[n]]))

for im in tqdm.tqdm(im_dict):
    if im_dict[im] is not None:
        if im_dict[im].ndim == 1:
            im_dict[im] = np.reshape(im_dict[im], (1,len(im_dict[im])))


im_keys = np.asarray(list(im_dict.keys()))

workers = np.asarray(dataDF['WorkerId'])
scoresDF1 = pd.DataFrame(index=range(len(im_keys)), columns=np.unique(workers).tolist())
scoresDF2 = pd.DataFrame(index=range(len(im_keys)), columns=np.unique(workers).tolist())





#
# Take the average of samples srcc scores between two
# groups of workers randomly split
#


def linfit(x, a, b):
    return a*x + b

import scipy.optimize as sco
def fit_lin(g1_scores, g2_scores, g1_scores_s2, g2_scores_s2):
    ind = np.argsort(g1_scores)
    x = g1_scores[ind]
    y = g2_scores[ind]

    ind_s2 = np.argsort(g1_scores_s2)
    x_s2 = g1_scores_s2[ind_s2]
    y_s2 = g2_scores_s2[ind_s2]

    popt, pcov = sco.curve_fit(linfit, x, y)
    popt_s2, pcov_s2 = sco.curve_fit(linfit, x_s2, y_s2)

    yhat = linfit(x,*popt)
    yhat_s2 = linfit(x_s2,*popt_s2)

    return popt,pcov, x, y, yhat, popt_s2,pcov_s2, x_s2, y_s2, yhat_s2




import math
epochs = 1

import numpy.random as npr
import scipy.stats

srcc1 = 0.0
srcc2 = 0.0
bins = 1
w_dict = {}
total_dict = {}
n=epochs
spearmans = np.zeros((n, 2))
popts = np.zeros((n,2))
popts_s2 = np.zeros((n,2))




for i in tqdm.tqdm(range(epochs)):

    w_dict = {}
    npr.shuffle(workers)
    g1 = workers[:len(workers)//2]
    g2 = workers[len(workers)//2:]
    
    for im in im_dict:
        g1_num = 0
        g2_num = 0
        g1_avg_s1 = 0
        g1_avg_s2 = 0
        g2_avg_s1 = 0
        g2_avg_s2 = 0
        
        total_num=0
        total_avg_s1=0
        total_avg_s2=0
        total_var_s1=0
        total_var_s2=0
        total_std_s1=0
        total_std_s2=0

        if im_dict[im] is None:
            continue
        for j in range(len(im_dict[im][:,0])):
            worker = im_dict[im][j,0]

            xn_s1 = int(im_dict[im][j,1])
            xn_s2 = int(im_dict[im][j,2])
            if(j!=0):
                u_n1_s1 = total_avg_s1/total_num
                u_n1_s2 = total_avg_s2/total_num
            elif(j==0):
                u_n1_s1 = 0 
                u_n1_s2 = 0 

            total_num+=1

            total_avg_s1+=int(im_dict[im][j,1])
            total_avg_s2+=int(im_dict[im][j,2])

            u_n_s1=total_avg_s1/total_num
            u_n_s2=total_avg_s2/total_num

            total_var_s1= total_var_s1+(xn_s1-u_n1_s1)*(xn_s1-u_n_s1)
            total_var_s2= total_var_s2+(xn_s2-u_n1_s2)*(xn_s1-u_n_s2)
            total_std_s1 = np.sqrt(total_var_s1/total_num)
            total_std_s2 = np.sqrt(total_var_s2/total_num)

            if (worker in g1):
                g1_num+=1
                g1_avg_s1+=int(im_dict[im][j,1])
                g1_avg_s2+=int(im_dict[im][j,2])
            else:
                g2_num+=1
                g2_avg_s1+=int(im_dict[im][j,1])
                g2_avg_s2+=int(im_dict[im][j,2])
        if ((g1_num>=10) and (g2_num>=10)):
           w_dict[im] = [(g1_avg_s1/g1_num)/bins, 
                         (g2_avg_s1/g2_num)/bins, 
                         (g1_avg_s2/g1_num)/bins, 
                         (g2_avg_s2/g2_num)/bins]
        if (total_num>=20):
            total_dict[im] =  [(total_avg_s1/total_num)/bins, 
                               (total_avg_s2/total_num)/bins,
                               (total_std_s1),
                               (total_std_s2)]


    g1s1 =[] 
    g2s1 =[] 
    g1s2 =[] 
    g2s2 =[] 
    total_s1=[]
    total_s2=[] 
    total_s1_std=[]
    total_s2_std=[] 
    total_names=[]
    
    for j in w_dict:
        g1s1.append(w_dict[j][0])
        g2s1.append(w_dict[j][1])
        g1s2.append(w_dict[j][2])
        g2s2.append(w_dict[j][3])
    for j in total_dict:
        total_names.append(j)
        total_s1.append(total_dict[j][0])
        total_s2.append(total_dict[j][1])
        total_s1_std.append(total_dict[j][2])
        total_s2_std.append(total_dict[j][3])
    

    srcc1+=scipy.stats.spearmanr(np.array(g1s1),np.array(g2s1))[0]
    srcc2+=scipy.stats.spearmanr(np.array(g1s2),np.array(g2s2))[0]

    g1s1 = np.array([g1s1]).reshape(-1)
    g1s2 = np.array([g1s2]).reshape(-1)
    g2s1 = np.array([g2s1]).reshape(-1)
    g2s2 = np.array([g2s2]).reshape(-1)

    total_names = np.array([total_names]).reshape(-1)
    total_s1 = np.array([total_s1]).reshape(-1)
    total_s2 = np.array([total_s2]).reshape(-1)
    total_s1_std = np.array([total_s1_std]).reshape(-1)
    total_s2_std = np.array([total_s2_std]).reshape(-1)


    popt,pcov, x, y, yhat, popt_s2,pcov_s2, x_s2, y_s2, yhat_s2 = fit_lin(g1s1,g2s1, g1s2, g2s2)
    popts[i] = popt
    popts_s2[i] = popt_s2


np.save('total_names', total_names)
np.save('total_s1', total_s1)
np.save('total_s2', total_s2)
np.save('total_s1_std', total_s1_std)
np.save('total_s2_std', total_s2_std)

srcc1 = srcc1/float(epochs)
srcc2 = srcc2/float(epochs)


