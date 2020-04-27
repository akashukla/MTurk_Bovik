# Standard imports
import re
import urllib
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Image lib imports
import cv2
from numpy.fft import fft2
from skimage.io import imread
from skimage.transform import resize

# Settings
mpl.rcParams['lines.markersize']=1.0

def image_to_ndarray(image_list, normalize):
    converted_images = []
    base = '../data/8k_data/'
    first_im = imread(base + image_list[0])
    first_im = resize(first_im, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
    first_im = first_im.astype('float32')

    mu = np.zeros(np.shape(first_im)) 

    for name in image_list:
        image_path = base + name
        image = imread(image_path)
        
        # Resizing image
        image = resize(image, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
        image = image.astype('float32')

        converted_images.append(image)
        mu += image

    mu = mu/len(image_list)
    if normalize: 
        for i in converted_images:
            i = i - mu

    return np.asarray(converted_images)

def image_features(image):
    l = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    l_max, l_min = np.max(l), np.min(l)
    l_avg = np.mean(l)
    l_dev = np.std(l)
    contrast = (l_max-l_min)/(l_max+l_min)

    rg = image[:,:,0]-image[:,:,1] # rg = r-g
    yb = 0.5*(image[:,:,0]+image[:,:,1]) - image[:,:,2] #yb = 0.5 (R+G) – B
    u_rg, sig_rg = np.mean(rg), np.std(rg)
    u_yb, sig_yb = np.mean(yb), np.std(yb) 
    colorfulness = np.sqrt(sig_rg**2+sig_yb**2) + 0.3*np.sqrt(u_rg**2 + u_yb**2)
    rms_cont = RMS_Contrast(image)

    return l_avg, l_dev, rms_cont, colorfulness



# Based on https://en.wikipedia.org/wiki/Contrast_(vision)
# Normalizes the image into float numbers [0,1]
# Puts image into grayscale
def RMS_Contrast(image):
    i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)/255
    M,N = i.shape
    i_avg = np.average(i)
    sum = 0
    for m in range(M):
        for n in range(N):
            sum += (i[m,n] - i_avg) ** 2
    return np.sqrt((1/M*N)*sum)

# Does not normalize the image. Image remains as ints
# Puts image into grayscale
def AVG_Luminance(image):
    return np.average(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


# Computers 2D FFT of the image
# Puts image into grayscale
def image_fft(image):
    return fft2(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


