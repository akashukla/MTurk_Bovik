import os, sys
import cv2
import urllib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# Inputs:
#   numpy ndarray with all data
#   dictionary with image results per workerID
#   list of all features needed to be visualized
#
# Visualizes data
def visualize_scores(data, im_dict, features):
    
    
