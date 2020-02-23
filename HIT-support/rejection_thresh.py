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

# Import golden image data
golden_resI = pd.read_csv('./dat/golden_results_old.csv')
golden_resII = pd.read_csv('./dat/golden_results_new.csv')

datI = np.asarray([golden_resI.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resI.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)
datII = np.asarray([golden_resII.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resII.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)

# Access golden image access methods
from goldenlib import *

# Averages and stdevs of all images
muI = np.mean(datI, axis=0)
muII = np.mean(datII, axis=0)

avg_stdevI = (1/len(datI))*np.sum(np.std(datI, axis=0))
avg_stdevII = (1/len(datII))*np.sum(np.std(datII, axis=0))

# Normalize subject ranges to expected values
diffI = [np.subtract(datI[i],muI) for i in range(len(datI))]
diffII = [np.subtract(datII[i],muII) for i in range(len(datII))]

norm_rangeI = np.max(abs_diffI, axis=1) - np.min(abs_diffI, axis=1)
norm_rangeII = np.max(abs_diffII, axis=1) - np.min(abs_diffII, axis=1)




