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

# Access golden image access methods
import pandas as pd
from goldenlib import *
import numpy.random as npr
from numpy.random import permutation, shuffle



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
contra_mu_names=['JPEGImages__2008_006461.jpg', 'VOC2012__2009_003324.jpg','JPEGImages__2011_004079.jpg','EMOTIC__COCO_val2014_000000147471.jpg','VOC2012__2011_003123.jpg']
#contra_mu_i = [0,8,49,64]
contra_mu_i=[np.argwhere(golden_names==contra_mu_names[i])[0,0] for i in range(len(contra_mu_names))]
#contra_mu = ordered_means[contra_mu_i]
contra_mu = golden_mu[contra_mu_i]
contra_std = golden_std[contra_mu_i]
avg_contra_corr = (1/len(golden_dat))*np.sum([sp.stats.spearmanr(contra_mu,golden_dat[i,contra_mu_i])[0] for i in range(len(golden_dat))]) 

srcc_thresh = avg_contra_corr

#
# Creating HIT list and hit_data.csv
#
images = np.genfromtxt('./dat/8k_image_names.csv','str',delimiter=',')[:,0]

# EDIT THESE TO CHANGE PARAMS
num_images = images.shape[0]            # total number of images in 8k dataset
set_size = 45                           # number of images per set
num_golden = 5                          # number of goldens per set
num_repeat = 5                          # number of repeats per set
data_points = 30                        # number of responses per image
num_total_sets = (num_images*data_points)/set_size   # total number of sets

rep_first_min = 0                       # Lower end of first repeated window
rep_first_max = 10                      # Upper end of first repeated window
rep_second_min = 20                     # Lower end of second repeated window
rep_second_max = 30                     # Upper end of second repeated window

ginx = [np.argwhere(images == golden_names[i])[0,0] for i in range(len(golden_names))]
images = images[np.delete(np.arange((images.shape[0])),ginx)]
num_images = images.shape[0]            # total number of images in 8k dataset

cnt_top = 0
cnt_eightk = 0

#hit_names=[]
#hit_list_ids=[]
final_data = []
ids=np.arange(num_images)
# Main HIT creation loop
while(cnt_top < num_total_sets):
    # Overall counter 
    cnt_top+=1
    # Resets at 8k
    cnt_eightk+=1
    if cnt_eightk*set_size+set_size > num_images:
        #last_used=cnt_eightk*set_size
        print('\n got to end, count*N is %d shuffling:'%cnt_eightk)
        #print('\n last image used was at index %d'%last_used)
        #print('\n there are %d images in total'%num_images)
        cnt_eightk=0
        # shuffle ids of the 8K bc we select 8K by ID
        shuffle(ids)
        print(ids)
    # HIT ids for next set
    hit_ids=ids[(cnt_eightk*set_size)%num_images:(cnt_eightk*set_size)%num_images+set_size]
    # now obsolete
    #hit_list_ids.append(hit_ids)
    # Local names for this set loop
    set_names=images[hit_ids]

    rep_ind_first = np.sort(np.random.choice(np.arange(rep_first_min,rep_first_max), num_repeat,replace=False))
    rep_names = set_names[rep_ind_first]
    rep_ind_second = np.random.choice(np.arange(rep_second_min,rep_second_max), num_repeat,replace=False)
    contra_mu_ind = np.sort(np.random.choice(np.arange(set_size), num_golden,replace=False))

    #Inserts Repeats
    for i in range(len(rep_ind_first)):
        # rep_ind_b = first ind of repeated
        toinsert=set_names[rep_ind_first[i]]
        set_names=np.insert(set_names,rep_ind_second[i],toinsert)

    # Inserts golden images
    for i in range(len(contra_mu_names)):
        set_names=np.insert(set_names,contra_mu_ind[i],contra_mu_names[i])


    contra_mu_ind = [np.argwhere(set_names==contra_mu_names[i])[0,0] for i in range(len(contra_mu_names))]
    rep_ind_first=[np.argwhere(set_names==rep_names[i])[0,0] for i in range(len(rep_names))]
    rep_ind_second=[np.argwhere(set_names==rep_names[i])[1,0] for i in range(len(rep_names))]
    # Here is what creates the csv file
    #hit_names.append(np.r_[set_size+num_golden+num_repeat, set_names,gic_ind,gsm_c.round().astype('int'), gss_c.round().astype('int'), rep_ind_b,rep_ind_e])
    set_write = np.r_[set_size+num_golden+num_repeat, set_names, num_golden, num_repeat, repeated_thresh, srcc_thresh, contra_mu_ind, contra_mu, contra_std, rep_ind_first, rep_ind_second]
    final_data.append(set_write)
 
np.savetxt('./gen/hit_data.csv',final_data,delimiter=',',fmt='%s')


    ### TODO: take the golden images we need to use



#
# Write final values to file
#

# Data Format
#   Number of images per HIT (set size)
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
#                       num_total_sets, 
#                       final_set, 
#                       num_golden, 
#                       num_repeat, 
#                       golden_dat,
#                       repeated_thresh,
#                       srcc_thresh,
#                       #golden_idx,
#                       #golden_means,
#                       #golden_std,
#                       #repeated_idx)
#np.savetxt('./gen/hit_data.csv',final_data,delimiter=',',fmt='%s')


def simulate():
    gic = np.random.choice(golden_used,6,replace=False)
    gic_ind = np.random.choice(np.arange(10),6,replace=False)
    gic_ind[4:]+=35
    gsm_c= np.array([gsm[golden_used==gic[i]][0] for i in range(len(gic))])
    gss_c= np.array([gss[golden_used==gic[i]][0] for i in range(len(gic))])
    amounts=np.zeros(18)
    for i in range(18):
        gsi2=gsi[:,gic_ind]
        indw=np.argwhere(np.absolute(gsi2[i]-np.mean(gsi2,axis=0)) > np.std(gsi2,axis=0))
        wrong=np.count_nonzero(indw)
        amounts[i]=(np.absolute(gsi2[i]-np.mean(gsi2,axis=0))/np.std(gsi2,axis=0) - 1)[indw].sum() 
    
        
        #print('i =', i, ', wrong:', wrong)
        #print('amount:',amounts[i])
    
    print('amounts', amounts, 'perfect ones', np.count_nonzero(amounts==0))
    return amounts
def get_people():
    amount_n=np.zeros((18,10))
    for n in range(10):
        amount_n[:,n] = simulate()
    #pavan=amount_n[0,:]    
    #dallas=amount_n[1,:]
    #mom=amount_n[4,:]
    #brain=amount_n[17,:]
    #print('\n pavan :',pavan)
    #print('\n dallas: ',dallas)
    #print('\n mom: ',mom)
    #print('\n brain: ',brain)

    #return pavan,dallas,mom,brain
    return amount_n


