#
# Import all needed libraries
#

#
# LIB IMPORTS
#
import os, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#
# Generate Data
#
from lib.dataload import *

csv_files = glob.glob('batch-data/batch*')
column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

# Make_predata method from lib/dataload.py returns a
# dictionary of images with the following structure:
# im_dict = {'image_name.jpg': ['workerid', 'slider1', slider2'
im_dict, data = make_predata(csv_files,column_headers)

workers = np.asarray(data['WorkerId'])
#workers = np.unique(workers)
epochs = 100

import numpy.random as npr
import scipy.stats

srcc1 = 0.0
srcc2 = 0.0

for i in range(epochs):
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
        
        if im_dict[im] is None:
            continue
        for j in range(len(im_dict[im][:,0])):
            worker = im_dict[im][j,0]
            if (worker in g1):
                g1_num+=1
                g1_avg_s1+=int(im_dict[im][j,1])
                g1_avg_s2+=int(im_dict[im][j,2])
            else:
                g2_num+=1
                g2_avg_s1+=int(im_dict[im][j,1])
                g2_avg_s2+=int(im_dict[im][j,2])
        if ((g1_num>=10) and (g2_num>=10)):
           w_dict[im] = [g1_avg_s1/g1_num, 
                         g2_avg_s1/g2_num, 
                         g1_avg_s2/g1_num, 
                         g2_avg_s2/g2_num]


    g1s1 =[] 
    g2s1 =[] 
    g1s2 =[] 
    g2s2 =[] 
    
    for j in w_dict:
        g1s1.append(w_dict[j][0])
        g2s1.append(w_dict[j][1])
        g1s2.append(w_dict[j][2])
        g2s2.append(w_dict[j][3])
    

    srcc1+=scipy.stats.spearmanr(np.array(g1s1),np.array(g2s1))[0]
    srcc2+=scipy.stats.spearmanr(np.array(g1s2),np.array(g2s2))[0]

srcc1 = srcc1/float(epochs)
srcc2 = srcc2/float(epochs)

import matplotlib.pyplot as plt
import matplotlib as mpl

x1 =[] 
y1 =[] 
x2 =[] 
y2 =[] 

for i in w_dict:
    x1.append(w_dict[i][0])
    y1.append(w_dict[i][1])
    x2.append(w_dict[i][2])
    y2.append(w_dict[i][3])

mpl.rcParams['lines.markersize'] = 1
plt.scatter(x1, y1) 
plt.show()

