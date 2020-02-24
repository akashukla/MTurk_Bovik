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

# Use Spearman's correlation on Golden Image Study
# participants to test 
# ONLY using Golden Image Study 2 from here on out
import scipy as sp
mu = muII
dat = datII
ordered_means_i = np.argsort(mu)
ordered_means = np.sort(mu)
avg_corr = (1/len(dat))*np.sum([sp.stats.spearmanr(ordered_means,dat[i,:])[0] for i in range(len(dat))]) 

# Select most contrasted mean values
contra_mu_i = [0,8,49,64]
contra_mu = ordered_means[contra_mu_i]
avg_contra_corr = (1/len(dat))*np.sum([sp.stats.spearmanr(contra_mu,dat[i,contra_mu_i])[0] for i in range(len(dat))]) 

# Access golden image access methods
from goldenlib import *





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

#
#np.savetxt('./gen/hit_data.csv',final_data,delimiter=',',fmt='%s')


