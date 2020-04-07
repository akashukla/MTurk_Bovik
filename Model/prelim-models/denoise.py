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
    # Take each worker and map out their 55 image scores with all the other
    # responses for those 55 images
    # Then hold out their score and calculate the mean for all 55 images
    # Order means from least to greatest
    # Take SRCC correlation with this worker with all 55 images and that 
    # Do this for all workers and rank them all by SRCC scores
    # Pick a reasonable threshold and eliminate all bad worker responses
    
    #workers = {keys: None for keys in np.asarray(data.loc[:,'WorkerId'])}
    #for im in im_dict:
    #    if im_dict[im] is not None:
    #        s1_mu = np.mean(im_dict[im][:,1].astype('int64'))
    #        s2_mu = np.mean(im_dict[im][:,2].astype('int64'))
    #        s1_std = np.std(im_dict[im][:,1].astype('int64'))
    #        s2_std = np.std(im_dict[im][:,2].astype('int64'))
    #        

def outlier_denoising(im_dict, data):
    # Take each worker and map out their 55 image scores with all the
    # other responses for those 55 images
    # Then hold out thier scores and calculate mean and std for all 55 images
    # Give worker a pentalty score according to how off from the std they are
    # Rank workers in terms of these penalty scores and choose elimination threshold
    # Eliminate worker responses that are above this elimination penalty score
