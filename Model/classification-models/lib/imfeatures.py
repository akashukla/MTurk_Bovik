import numpy as np
from numpy.fft import fft2
import os,sys,cv2

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


