#
# LIB IMPORTS
#

import os, sys
import numpy as np
import pandas as pd


csv_files = ['lib/batch-data/batch1_results.csv',
             'lib/batch-data/batch2_results.csv',
             'lib/batch-data/batch3_results.csv',
             'lib/batch-data/batch4_results.csv',
             'lib/batch-data/batch5_results.csv',
             'lib/batch-data/batch6_results.csv']

column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

im_dict, data = make_predata(csv_files,column_headers)

# Choose a split for workers
    # Find all overlapping images
    # Find mean score for each group for each image
    # Correlate them
# Average the correlations for all the splits
