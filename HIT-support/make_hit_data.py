#
# This program is designed to calculate the 
# threshold for repeated images as well
# as the SROCC values and threshold that we
# will be using for the HIT study 
# in-session rejection
#
# For bugs contact Author: MK Swaminathan
#
import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# Import golden image data
#
golden_resI = pd.read_csv('./dat/golden_results_old.csv')
golden_resII = pd.read_csv('./dat/golden_results_new.csv')

datI = np.asarray([golden_resI.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resI.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)
datII = np.asarray([golden_resII.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resII.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)



#
# Calculate thresholds
#

# Averages and stdevs of all images
muI = np.mean(datI, axis=0)
muII = np.mean(datII, axis=0)

avg_stdevI = (1/len(datI))*np.sum(np.std(datI, axis=0))
avg_stdevII = (1/len(datII))*np.sum(np.std(datII, axis=0))

# Normalize subject ranges to expected values
diffI = [np.subtract(datI[i],muI) for i in range(len(datI))]
diffII = [np.subtract(datII[i],muII) for i in range(len(datII))]

norm_rangeI = np.max(diffI, axis=1) - np.min(diffI, axis=1)
norm_rangeII = np.max(diffII, axis=1) - np.min(diffII, axis=1)


#
# Use Spearman's correlation on Golden Image Study
# participants to test 
# ONLY using Golden Image Study 2 from here on out
#
import scipy as sp
golden_dat = np.delete(datII,17,axis=0) # Removing the really terrible outlier
golden_mu = np.mean(golden_dat, axis=0)
golden_std = np.mean(golden_dat, axis=0)

repeated_thresh = (1/len(golden_dat))*np.sum(np.std(golden_dat, axis=0))

ordered_means_i = np.argsort(golden_mu)
ordered_means = np.sort(golden_mu)
avg_corr = (1/len(golden_dat))*np.sum([sp.stats.spearmanr(ordered_means,golden_dat[i,:])[0] for i in range(len(golden_dat))]) 

# Select most contrasted mean values
contra_mu_i = [0,8,49,64]
contra_mu = ordered_means[contra_mu_i]
avg_contra_corr = (1/len(golden_dat))*np.sum([sp.stats.spearmanr(contra_mu,golden_dat[i,contra_mu_i])[0] for i in range(len(golden_dat))]) 

srcc_thresh = avg_contra_corr

#
# Creating HIT list and hit_data.csv
#

# Access golden image access methods
import pandas as pd
from goldenlib import *
import numpy.random as npr
from numpy.random import permutation, shuffle

images = np.genfromtxt('./dat/8k_image_names.csv','str',delimiter=',')[:,0]

# EDIT THESE TO CHANGE PARAMS
num_images = images.shape[0]            # total number of images in 8k dataset
set_size = 43                           # number of images per set
num_golden = 5                          # number of goldens per set
num_repeat = 5                          # number of repeats per set
data_points = 30                        # number of responses per image
num_total_sets = 
    (num_images*data_points)/set_size   # total number of sets

rep_first_min = 0                       # Lower end of first repeated window
rep_first_max = 10                      # Upper end of first repeated window
rep_second_min = 20                     # Lower end of second repeated window
rep_second_max = 30                     # Upper end of second repeated window

ginx = [np.argwhere(images == golden_names[i])[0,0] for i in range(len(golden_names))]
images = images[np.delete(np.arange((images.shape[0])),ginx)]

cnt_top = 0
cnt_eightk = 0

# Main HIT creation loop
while(cnt_top < ):
    # Overall counter 
    cnt_top+=1
    # Resets at 8k
    cnt_eightk+=1

    ### TODO: take the golden images we need to use



#
# Write final values to file
#

# Data Format
#   Number of images per HIT (set size)
#   Number of total sets
#   Image names 
#   Number of golden images
#   Number of repeated images
#   Repeated threshold
#   Spearman threshold
#   Golden image indeces
#   Golden image mean values
#   Golden image stdev
#   Repeated image indeces (2X number of repeated)
#       (first occurances, second occurances)

#final_data = np.array(set_size, 
                       num_total_sets, 
                       final_set, 
                       num_golden, 
                       num_repeat, 
                       golden_dat,
                       repeated_thresh,
                       srcc_thresh,
                       #golden_idx,
                       #golden_means,
                       #golden_std,
                       #repeated_idx)
#np.savetxt('./gen/hit_data.csv',final_data,delimiter=',',fmt='%s')


