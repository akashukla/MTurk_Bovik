import re
import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd

#
# Inputs:
#   numpy ndarray with all data
#   dictionary with image results per workerID
#
# Output:
#   cleaned data
#
def correlation_denoising(im_dict, data):
    pass
    # Take each worker and map out their 55 image scores with all the other
    # responses for those 55 images
    # Then hold out their score and calculate the mean for all 55 images
    # Order means from least to greatest
    # Take SRCC correlation with this worker with all 55 images and that 
    # Do this for all workers and rank them all by SRCC scores
    # Pick a reasonable threshold and eliminate all bad worker responses
    
def outlier_denoising(im_dict, data):
    # Take each worker and map out their 55 image scores with all the
    # other responses for those 55 images
    # Then hold out thier scores and calculate mean and std for all 55 images
    # Give worker a pentalty score according to how off from the std they are
    # Rank workers in terms of these penalty scores and choose elimination threshold
    # Eliminate worker responses that are above this elimination penalty score

    workers = {keys: 0 for keys in np.asarray(data.loc[:,'WorkerId'])}
    for im in im_dict:
        if im_dict[im] is not None:
            exception = False
            if len(im_dict[im][:,0]) <= 2:
                exception = True
            for worker_i in range(len(im_dict[im][:,0])):
                if exception:
                    break
                worker_s1 = float(im_dict[im][worker_i,1])
                worker_s2 = float(im_dict[im][worker_i,2])

                s1_vals = [score for score in im_dict[im][:,1]]
                s1_vals.pop(worker_i)
                s1_vals = np.asarray(s1_vals).astype('int64')
                
                s2_vals = [score for score in im_dict[im][:,2]]
                s2_vals.pop(worker_i)
                s2_vals = np.asarray(s2_vals).astype('int64')

                s1_mu = np.mean(s1_vals)
                s2_mu = np.mean(s2_vals)
                s1_std = np.std(s1_vals)
                s2_std = np.std(s2_vals)
                
                # Penalty is proportional to how many std they are from the mean
                if abs(float(worker_s1) - s1_mu) > s1_std and not exception and s1_std != 0:
                    s1_penalty = abs(worker_s1 - s1_mu)/s1_std
                else:
                    s1_penalty = 0.0
                if abs(float(worker_s2) - s2_mu) > s2_std and not exception and s2_std != 0:
                    s2_penalty = abs(worker_s2 - s2_mu)/s2_std
                else:
                    s2_penalty = 0.0

                workers[im_dict[im][worker_i,0]] += (s1_penalty + s2_penalty)
                
    return workers        

            


