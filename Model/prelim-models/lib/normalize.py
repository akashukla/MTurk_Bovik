import cv2
import numpy as np

def zero_center(Images):
    mu = np.zeros(Images[0].shape)
    for image in Images:
        mu = mu + image
    mu = mu/Images.shape[0]

    for i in range(Images.shape[0]):
        Images[i] = Images[i] - mu

    return Images
