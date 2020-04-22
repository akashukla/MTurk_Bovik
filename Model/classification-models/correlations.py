#
# Import all needed libraries
#

#
# LIB IMPORTS
#
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#
# Generate Data
#
from lib.dataload import *
from lib.denoise import *
from lib.normalize import *

csv_files = ['lib/batch-data/batch1_results.csv',
             'lib/batch-data/batch2_results.csv',
             'lib/batch-data/batch3_results.csv',
             'lib/batch-data/batch4_results.csv',
             'lib/batch-data/batch5_results.csv',
             'lib/batch-data/batch6_results.csv',
             'lib/batch-data/batch7_results.csv',
             'lib/batch-data/batch8_results.csv',
             'lib/batch-data/batch9_results.csv',
             'lib/batch-data/batch10_results.csv',
             'lib/batch-data/batch12_results.csv',
             'lib/batch-data/batch13_results.csv',
             'lib/batch-data/batch14_results.csv',
             'lib/batch-data/batch15_results.csv',
             'lib/batch-data/batch16_results.csv',
             'lib/batch-data/batch17_results.csv',
             'lib/batch-data/batch18_results.csv',
             'lib/batch-data/batch19_results.csv',
             'lib/batch-data/batch20_results.csv',
             'lib/batch-data/batch21_results.csv',
             'lib/batch-data/batch22_results.csv',
             'lib/batch-data/batch23_results.csv',
             'lib/batch-data/batch24_results.csv',
             'lib/batch-data/batch25_results.csv',
             'lib/batch-data/batch26_results.csv']

column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

# Make the data
im_dict, data = make_predata(csv_files,column_headers)



