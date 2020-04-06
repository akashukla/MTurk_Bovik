import re
import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd


#
# Inputs:
#   list of csv files
#   list of column headers
#
# Returns:
#   numpy ndarray with all data
#   dictionary with image results per workerID
#

def make_predata(csv_files, column_headers):
    dataDF = []
    for f in csv_files:
        df = pd.read_csv(f)
        df = df[df.loc[:,'AssignmentStatus'] == 'Approved']
        df = df[df.loc[:,'Answer.set_number'] != 'initial']
        df = df.loc[:,column_headers] 
        if re.match(r'.*batch[1-3]', f) is None:
            df['HitDataFile'] = np.repeat(0,len(df))
        else: 
            df['HitDataFile'] = np.repeat(1,len(df))
        dataDF.append(df) 
    
    data = pd.concat(dataDF,ignore_index=True)
    
    hit_data_orig = np.genfromtxt('batch-data/hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]
    #hit_data = np.genfromtxt('batch-data/hit_data.csv',delimiter=',',dtype='str')[:,1:56]
    im_names = pd.read_csv('../data/8k_image_names.csv', 
                              header=None, 
                              usecols=[0])
    im_names = (im_names.loc[:,0]).tolist()
    im_dict = {keys: None for keys in im_names}
    
    for w in range(len(data)):
        worker_id = data.loc[w,'WorkerId']
        set_num = int(data.loc[w,'Answer.set_number'])
        slider1_vals = np.asarray(data.loc[w,'Answer.slider_values'].split(','))[:-1].astype('int64')
        slider2_vals = np.asarray(data.loc[w,'Answer.slider_values2'].split(','))[:-1].astype('int64')
       
        #if int(data.loc[w,'HitDataFile']) == 0:
        set_names = hit_data_orig[set_num-1]
        #else:
        #    set_names = hit_data[set_num-1]

        for n in range(len(set_names)):
            if im_dict[set_names[n]] is None:
                im_dict[set_names[n]] = np.asarray([worker_id,
                                                    slider1_vals[n],
                                                    slider2_vals[n]])
            else:
                im_dict[set_names[n]] = np.vstack((im_dict[set_names[n]],
                                                  [worker_id,
                                                   slider1_vals[n],
                                                   slider2_vals[n]]))
    for im in im_dict:
        if im_dict[im] is not None: 
            if im_dict[im].ndim == 1:
                im_dict[im] = np.reshape(im_dict[im],(1,len(im_dict[im])))
                

    return im_dict, data

#csv_files = ['batch-data/batch1_results.csv','batch-data/batch2_results.csv','batch-data/batch3_results.csv']
#column_headers = ['AssignmentStatus','Answer.set_number','WorkerId','Answer.slider_values','Answer.slider_values2']
#
#im_dict, data = make_predata(csv_files,column_headers)


