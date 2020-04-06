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
def denoise(im_dict, data):
    # Create a dictionary of workers
    workers = {keys: None for keys in np.asarray(data.loc[:,'WorkerId'])}
    # iteratively go through and search for outliers and assign them penalty
    for im in im_dict:
        if im_dict[im] is not None:
            s1_mu = np.mean(im_dict[im][:,1].astype('int64'))
            s2_mu = np.mean(im_dict[im][:,2].astype('int64'))
            s1_std = np.std(im_dict[im][:,1].astype('int64'))
            s2_std = np.std(im_dict[im][:,2].astype('int64'))

