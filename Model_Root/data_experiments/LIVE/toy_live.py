# Author: Meenakshi Swaminathan
# 
# Experimenting with LIVE Subjective Image Quality Database
#

import os, os.path
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.io import loadmat

data_release2 = loadmat('./dmos_realigned.mat')
print (data_release2)


# MATLAB mat files
# The file dmos.mat has two arrays of length 982 each: dmos and orgs. orgs(i)==0 for distorted images.
# The arrays dmos and orgs are arranged by concatenating the dmos (and orgs) variables
# for each database as follows:
# 
# dmos=[dmos_jpeg2000(1:227) dmos_jpeg(1:233) white_noise(1:174) gaussian_blur(1:174) fast_fading(1:174)] where
# dmos_distortion(i) is the dmos value for image "distortion\imgi.bmp" where distortion can be one of the five
# described above.
# 
# The values of dmos when corresponding orgs==1 are zero (they are reference images). Note that imperceptible
# loss of quality does not necessarily mean a dmos value of zero due to the nature of the score processing used.
# 
# The file refnames_all.mat contains a cell array refnames_all. Entry refnames_all{i} is the name of
# the reference image for image i whose dmos value is given by dmos(i). If orgs(i)==0, then this is a valid
# dmos entry. Else if orgs(i)==1 then image i denotes a copy of the reference image. The reference images are
# provided in the folder refimgs.


