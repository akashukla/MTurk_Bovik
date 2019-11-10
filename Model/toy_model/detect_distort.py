import scipy.io 
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

#
# Import LIVE Image Data
#
import re
from os import listdir
from os.path import isfile, join

head = '../../LIVE/'
paths = [d+'/' for d in listdir(head) if d in ['jpeg','jp2k', 'refimgs']]
dist_images = []
refimages = []

for path in paths:
    for f in listdir(head+path):
        if re._compile('.*?bmp',flags=0).match(f):
            dist_images.append(head+path+f)
            if path=='refimgs/': refimages.append(f)
dist_images = np.asarray(dist_images)

#
# The Question:
# Can this learn what it means to be a high
# quality image? 
#

import cv2
import torch
from torchvision.transforms import *
