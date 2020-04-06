#
# The purpose of this script is to 
# make it ammenable to visually 
# inspect the data from the 
# golden image study.
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
from goldenlib import *

# Import golden image data
golden_resI = pd.read_csv('./dat/golden_results_old.csv')
golden_resII = pd.read_csv('./dat/golden_results_new.csv')

datI = np.asarray([golden_resI.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resI.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)
datII = np.asarray([golden_resII.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resII.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)

# Order by variances
varI_i = np.argsort(np.var(datI, axis=0))
varII_i = np.argsort(np.var(datII, axis=0))




# Plot to inspect spread of data overall and expected value
for i in range(len(datI)):
    plt.scatter(np.arange(len(datI[0])), datI[i,varI_i])

plt.xticks(np.arange(len(datI[0])), varI_i)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting Golden Image Data OLD by Sorted Variances')
plt.tight_layout()

plt.figure()

for i in range(len(datII)):
    plt.scatter(np.arange(len(datII[0])), datII[i,varII_i])

plt.xticks(np.arange(len(datII[0])), varII_i)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting Golden Image Data NEW by Sorted Variances')
plt.tight_layout()

plt.figure()




# Plotting Expected Values
plt.errorbar(np.arange(len(datI[0])), np.mean(datI,axis=0)[varI_i], np.std(datI, axis=0)[varI_i], fmt='o')
plt.xticks(np.arange(len(datI[0])), varI_i)
plt.ylim(0,50)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting OLD Expected Value by Sorted Variance')
plt.tight_layout()

plt.figure()

plt.errorbar(np.arange(len(datII[0])), np.mean(datII,axis=0)[varII_i], np.std(datII, axis=0)[varII_i], fmt='o')
plt.xticks(np.arange(len(datII[0])), varII_i)
plt.ylim(0,50)
plt.xlabel('Sorted Variance')
plt.ylabel('Compression Param')
plt.title('Plotting NEW Expected Value by Sorted Variance')
plt.tight_layout()

#plt.figure()


#for i in range(len(datI)):
#    plt.scatter(np.arange(len(datI[0])), datI[i,varI_i],marker='x')
#
#plt.errorbar(np.arange(len(datI[0])), np.mean(datI,axis=0)[var_i], 1.5*np.std(datI, axis=0)[varI_i], fmt='o')
#
#plt.show()

