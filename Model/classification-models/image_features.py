import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import matplotlib as mpl
mpl.rcParams['lines.markersize']=1.0

def gen_batch_data():
    batch_files=glob.glob('batch-data/batch*')
    hit_files=glob.glob('batch-data/hit*')
    batch_dfs = []
    batch_setnums = []
    batch_svals = []
    batch_svals2 = []
    
    batch_svals_arr=np.array([], dtype='float64')
    batch_svals2_arr=np.array([], dtype='float64')
    batch_setnums_arr=np.array([], dtype='int')
    batch_image_names_arr= np.array([], dtype='str')
    batch_workers_arr= np.array([], dtype='str')
    
    hit_data_orig = np.genfromtxt('batch-data/hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data_2_electric_boogaloo= np.genfromtxt('batch-data/hit_data_2_electric_boogaloo.csv',delimiter=',',dtype='str')[:,1:56]
    hit_data = np.genfromtxt('batch-data/hit_data.csv',delimiter=',',dtype='str')[:,1:56]
    
    hit_data_orig_batches = np.arange(1,4)
    hit_data_2_electric_boogaloo_batches = np.arange(4,7)
    hit_data_batches = np.arange(17,27)
    
    orig_ind = [11, 19, 24]
    electric_ind = [10, 17, 22]
    
    for i in range(len(batch_files)):
        batch_dfs.append(pd.read_csv(batch_files[i]))
        batch_dfs[i] = batch_dfs[i][batch_dfs[i].loc[:,'AssignmentStatus']=='Approved']
        batch_dfs[i] = batch_dfs[i][batch_dfs[i].loc[:,'Answer.set_number']!='initial']
    
        batch_setnums.append(batch_dfs[i].loc[:,'Answer.set_number'].values.astype('int'))
    
        #batch_svals.append([np.fromstring(np.array(batch_dfs[i].loc[:,'Answer.slider_values'])[j],dtype='int',sep=',') for j in range(len(batch_dfs[i]))])
        batch_svals.append([np.fromstring(batch_dfs[i].loc[:,'Answer.slider_values'].values[j],dtype='int',sep=',') for j in range(len(batch_dfs[i]))])
        batch_svals2.append([np.fromstring(batch_dfs[i].loc[:,'Answer.slider_values2'].values[j],dtype='int',sep=',') for j in range(len(batch_dfs[i]))])
    
        batch_svals_arr = np.append(batch_svals_arr,np.array(batch_svals[i]))
        batch_svals2_arr = np.append(batch_svals2_arr,np.array(batch_svals2[i]))
        batch_setnums_arr = np.append(batch_setnums_arr,batch_setnums[i])
        batch_workers_arr= np.append(batch_workers_arr,np.repeat(batch_dfs[i].loc[:,'WorkerId'].values.astype('str'), 55) )
    
        if i in orig_ind:
            batch_image_names_arr = np.append(batch_image_names_arr,hit_data_orig[batch_setnums[i]%len(hit_data_orig)-1])
            #print(i, ': orig')
    
        elif i in electric_ind:
            batch_image_names_arr = np.append(batch_image_names_arr,hit_data_2_electric_boogaloo[batch_setnums[i]%len(hit_data_2_electric_boogaloo)-1])
            #print(i, ': electric')
        else:
            batch_image_names_arr = np.append(batch_image_names_arr,hit_data[batch_setnums[i]%len(hit_data)-1])
            #print(i, ': hit_data')
    
    
    #result is batch_image_names_arr, batch_svals_arr,batch_svals2_arr, batch_workers_arr
    np.save('batch_image_names_arr', batch_image_names_arr)
    np.save('batch_svals_arr', batch_svals_arr)
    np.save('batch_svals2_arr', batch_svals2_arr)
    np.save('batch_workers_arr', batch_workers_arr)
    return batch_image_names_arr, batch_svals_arr, batch_svals2_arr, batch_workers_arr

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
    rms_cont = rms_contrast(image)

    return l_avg, l_dev, rms_cont, colorfulness

def rms_contrast(image):
    i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)/255
    M,N = i.shape
    i_avg = np.average(i)
    sum = 0
    for m in range(M):
        for n in range(N):
            sum += (i[m,n] - i_avg) ** 2
    return np.sqrt((1/M*N)*sum)




#g1_s1 = np.zeros(uq_image.shape[0])
#g2_s1 = np.zeros(uq_image.shape[0])
#g1_s2 = np.zeros(uq_image.shape[0])
#g2_s2 = np.zeros(uq_image.shape[0])
#w = np.zeros(uq_image.shape[0])


image_names = np.load('batch_image_names_arr.npy')
svals = np.load('batch_svals_arr.npy')
svals2 = np.load('batch_svals2_arr.npy')
workers = np.load('batch_workers_arr.npy')

uq_image, uq_ind, uq_count = np.unique(image_names,return_index=True, return_counts=True)


image_svals = np.zeros((uq_image.shape[0], 2), dtype='float64')

##   image_feats = np.zeros((uq_image.shape[0], 4),dtype='float64')
def gen_svals_features():
    i=0
    for uqi in uq_image:
        if i%100==0:
            print(i)
        image_svals[i]= np.array([np.mean(svals[image_names==uqi]), np.mean(svals2[image_names==uqi])])
    ##       #sv1 = batch_svals_arr[batch_image_names_arr==uqi] 
    ##       #sv2 = batch_svals2_arr[batch_image_names_arr==uqi]
    ##       ##w[i] = batch_workers_arr[batch_image_names_arr==uqi]
    ##       ##w[i] = np._unique(batch_workers_arr[i])
    ##       ##w1=w[i][:len(w[i]//2)]
    ##       ##w2=w[i][len(w[i]//2):]
    ##       #g1_s1[i] = sv1[0:sv1.shape[0]//2].mean()
    ##       #g2_s1[i] = sv1[sv1.shape[0]//2:].mean()
    ##       #g1_s2[i] = sv2[0:sv2.shape[0]//2].mean()
    ##       #g2_s2[i] = sv2[sv2.shape[0]//2:].mean()
    
        imi=cv2.imread('../data/8k_data/'+uqi)
        image_feats[i] = image_features(imi)
        image_svals
        i+=1

    np.save('image_feats', image_feats)
    np.save('image_svals', image_svals)

image_feats=np.load('image_feats.npy')
image_svals=np.load('image_svals.npy')
