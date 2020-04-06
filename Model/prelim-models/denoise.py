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
       
