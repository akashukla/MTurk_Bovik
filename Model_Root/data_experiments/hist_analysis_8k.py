# Author: Meenakshi Swaminathan
# 
# This program runs some experiments on the 8K images
# given to our group by Professor Bovik and Zhenqiang
# that are statistically similar to social media 
# images
#

import os, os.path
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_max_intensity = 256

def histogram(I, nBins):
    """ This function takes in an image and returns the levels
        and bins associated with its histogram.
    """
    levels = [(I >= (i*(_max_intensity/nBins))).sum() - (I >= ((i+1)*(_max_intensity/nBins))).sum() for i in range(nBins)]
    bins = [i*(256/nBins) for i in range(nBins)]
    return np.asarray(levels), np.asarray(bins)

def average_histogram(Images, nBins):
    """ This function takes in a list of images and returns the levels
        and bins associated with its average histogram.
    """
    levels, bins = np.zeros(_max_intensity), np.zeros(_max_intensity)
    first = True
    for i in Images:
        l, b = histogram(i, _max_intensity)
        levels+=l
        if first: bins = b
        first = False
    levels/=len(Images)
    return levels, bins   

Images = []
path = '../datasets/Zhenqiang_8K_Images'
sizes = [1, 10, 100, 500, 1000, 8000]
for s in sizes:
    count = 0
    for f in os.listdir(path):
        if count == s: break
        Images.append(plt.imread(os.path.join(path,f)))
        count+=1

    levels, bins = average_histogram(Images,_max_intensity)

    plt.figure()
    plt.bar(bins,levels,width=1)
    plt.savefig('./gen/'+str(s)+'_hist.png')

