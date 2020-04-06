import re
import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd

from dataload import *
        
csv_files = ['batch-data/batch1_results.csv','batch-data/batch2_results.csv','batch-data/batch3_results.csv']
column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']

im_dict, data = make_predata(csv_files,column_headers)



