# Standard imports
import re
import urllib
import os, sys
import numpy as np
import pandas as pd

# Image lib imports
import cv2
from skimage.io import imread
from skimage.transform import resize


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
        if re.match(r'.*batch[1-3]', f) is not None:
            df['HitDataFile'] = np.repeat(0,len(df))
        elif re.match(r'.*batch[4-6]',f) is not None: 
            df['HitDataFile'] = np.repeat(1,len(df))
        else:
            df['HitDataFile'] = np.repeat(2,len(df))
        dataDF.append(df) 
    
    data = pd.concat(dataDF,ignore_index=True)

    hit_data_orig = np.genfromtxt('lib/batch-data/hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data_electric_boogaloo = np.genfromtxt('lib/batch-data/hit_data_2_electric_boogaloo.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data = np.genfromtxt('lib/batch-data/hit_data.csv',delimiter=',',dtype='str')[:,1:56]
    
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
       
        if int(data.loc[w,'HitDataFile']) == 0:
            set_names = hit_data_orig[set_num-1]
        elif int(data.loc[w,'HitDataFile']) == 1:
            set_names = hit_data_electric_boogaloo[set_num-1]
        else:
            set_names = hit_data[set_num-1]
        
        
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


#
# Input: 
#   dataframe with cleaned data
#
# Output:
#   image arrays and scores to train on
#
def format_data(data):
    hit_data_orig = np.genfromtxt('lib/batch-data/hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data_electric_boogaloo = np.genfromtxt('lib/batch-data/hit_data_2_electric_boogaloo.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data = np.genfromtxt('lib/batch-data/hit_data.csv',delimiter=',',dtype='str')[:,1:56]
    

    im_names = pd.read_csv('../data/8k_image_names.csv', 
                              header=None, 
                              usecols=[0])
    im_names = (im_names.loc[:,0]).tolist()
    im_dict = {keys: None for keys in im_names}
    scores1_dict = {}
    scores2_dict = {}
 
    for w in range(len(data)):
        set_num = int(data.loc[w,'Answer.set_number'])
        slider1_vals = np.asarray(data.loc[w,'Answer.slider_values'].split(','))[:-1].astype('int64')
        slider2_vals = np.asarray(data.loc[w,'Answer.slider_values2'].split(','))[:-1].astype('int64')
      
        if int(data.loc[w,'HitDataFile']) == 0:
            set_names = hit_data_orig[set_num-1]
        elif int(data.loc[w,'HitDataFile']) == 1:
            set_names = hit_data_electric_boogaloo[set_num-1]
        else:
            set_names = hit_data[set_num-1]
        
        for n in range(len(set_names)):
            if im_dict[set_names[n]] is None:
                im_dict[set_names[n]] = np.asarray([slider1_vals[n],
                                                    slider2_vals[n]])
            else:
                im_dict[set_names[n]] = np.vstack((im_dict[set_names[n]],
                                                  [slider1_vals[n],
                                                   slider2_vals[n]]))

    # Ensure consistent format
    for im in im_dict:
        if im_dict[im] is not None: 
            if im_dict[im].ndim == 1:
                im_dict[im] = np.reshape(im_dict[im],(1,len(im_dict[im])))

    # Set dicts for images - s1 and s2 scores
    for im in im_dict:
        if im_dict[im] is not None:
           s1_vals = [score for score in im_dict[im][:,0]]
           s1_vals = np.asarray(s1_vals).astype('int64')
           
           s2_vals = [score for score in im_dict[im][:,1]]
           s2_vals = np.asarray(s2_vals).astype('int64')

           scores1_dict[im] = np.mean(s1_vals)
           scores2_dict[im] = np.mean(s2_vals)
    
    # Convert to numpy arrays
    images_s1 = np.asarray(list(scores1_dict.keys()))
    images_s2 = np.asarray(list(scores2_dict.keys()))

    images_arr_s1 = image_to_ndarray(images_s1)
    # images_arr_s2 = image_to_ndarray(images_s2)
    scores_s1 = np.asarray(list(scores1_dict.values())).astype('float32')
    scores_s2 = np.asarray(list(scores2_dict.values())).astype('float32')

    # Image arrays 1 and 2 end up being the same
    return images_arr_s1, images_s1, scores_s1, scores_s2

def image_to_ndarray(image_list):
    converted_images = []
    for name in image_list:
        image_path = '../data/8k_data/' + name
        image = imread(image_path)
        
        # Resizing image
        image = resize(image, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
        image = image.astype('float32')

        converted_images.append(image)

    return np.asarray(converted_images)




