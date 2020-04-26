#
# Import all needed libraries
#

#
# LIB IMPORTS
#
import os, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#
# Generate Data
#
from lib.dataload import *

csv_files = glob.glob('lib/batch-data/batch*')
column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

# Make_predata method from lib/dataload.py returns im_dict and data
# data is a data frame contaning the data of the column headers above
#
# im _dict is a dictionary of images with the following structure:
# im_dict = {'image_name.jpg': ['WorkerId', 'slider1', slider2'],
#            'another_im.jpg': ['ASBCJEFJ', '34', '21'],
#               ... etc. }
im_dict, data = make_predata(csv_files,column_headers)



