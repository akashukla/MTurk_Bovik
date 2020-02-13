#
# The purpose of this script is to 
# make it ammenable to visually 
# inspect the data from the 
# golden image study.
#
# Author: MK Swaminathan
#

import os, sys
import cv2
import urllib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# URL to IM Lib
from golden_im import *

# Import golden image data
golden_res = pd.read_csv('./golden_results.csv')

dat = np.asarray([golden_res.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_res.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)

# Order by variances
var_i = np.argsort(np.var(dat, axis=0))




# Plot to inspect spread of data overall and expected value
for i in range(len(dat)):
    plt.scatter(np.arange(len(dat[0])), dat[i,var_i])

plt.xticks(np.arange(len(dat[0])), var_i)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting Golden Image Data by Sorted Variances')
plt.tight_layout()

plt.figure()

plt.errorbar(np.arange(len(dat[0])), np.mean(dat,axis=0)[var_i], np.std(dat, axis=0)[var_i], fmt='o')
plt.xticks(np.arange(len(dat[0])), var_i)
plt.ylim(0,50)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting Expected Value by Sorted Variance')
plt.tight_layout()

plt.figure()
for i in range(len(dat)):
    plt.scatter(np.arange(len(dat[0])), dat[i,var_i],marker='x')

plt.errorbar(np.arange(len(dat[0])), np.mean(dat,axis=0)[var_i], 1.5*np.std(dat, axis=0)[var_i], fmt='o')

plt.show()

