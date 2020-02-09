# coding: utf-8
import gluoncv
from gluoncv import utils
import numpy as np
import numpy.random as npr
from numpy.random import permutation, shuffle
import os, csv
import numpy as np
from array import *
import pandas as pd

N = 44
num_golden = 5
count=np.r_[0,0]
names=np.genfromtxt('8k_image_names.csv','str', delimiter=',')
names=names[:,0]

golden_names = np.array(['EMOTIC__COCO_train2014_000000208055.jpg', 'EMOTIC__COCO_train2014_000000211272.jpg', 'EMOTIC__COCO_train2014_000000211486.jpg', 'EMOTIC__COCO_train2014_000000212083.jpg', 'EMOTIC__COCO_train2014_000000221881.jpg', 'EMOTIC__COCO_train2014_000000222140.jpg', 'EMOTIC__COCO_train2014_000000223888.jpg', 'EMOTIC__COCO_train2014_000000226597.jpg', 'EMOTIC__COCO_train2014_000000229188.jpg', 'EMOTIC__COCO_train2014_000000230433.jpg', 'EMOTIC__COCO_train2014_000000239307.jpg', 'EMOTIC__COCO_train2014_000000241421.jpg', 'EMOTIC__COCO_train2014_000000250357.jpg', 'EMOTIC__COCO_train2014_000000257297.jpg', 'EMOTIC__COCO_train2014_000000265100.jpg', 'EMOTIC__COCO_train2014_000000266479.jpg', 'EMOTIC__COCO_train2014_000000268209.jpg', 'EMOTIC__COCO_train2014_000000276128.jpg', 'EMOTIC__COCO_train2014_000000277122.jpg', 'EMOTIC__COCO_train2014_000000297359.jpg', 'EMOTIC__COCO_train2014_000000300598.jpg', 'EMOTIC__COCO_train2014_000000301855.jpg', 'EMOTIC__COCO_train2014_000000302102.jpg', 'EMOTIC__COCO_train2014_000000304548.jpg', 'EMOTIC__COCO_train2014_000000305105.jpg', 'EMOTIC__COCO_train2014_000000307894.jpg', 'EMOTIC__COCO_train2014_000000308353.jpg', 'EMOTIC__COCO_train2014_000000311706.jpg', 'EMOTIC__COCO_train2014_000000318496.jpg', 'EMOTIC__COCO_train2014_000000319690.jpg', 'EMOTIC__COCO_train2014_000000319905.jpg', 'EMOTIC__COCO_train2014_000000322212.jpg', 'EMOTIC__COCO_train2014_000000325981.jpg', 'EMOTIC__COCO_train2014_000000326504.jpg', 'EMOTIC__COCO_train2014_000000327810.jpg', 'EMOTIC__COCO_train2014_000000329587.jpg', 'EMOTIC__COCO_train2014_000000329942.jpg', 'EMOTIC__COCO_train2014_000000334338.jpg', 'EMOTIC__COCO_train2014_000000341623.jpg', 'EMOTIC__COCO_train2014_000000341905.jpg', 'EMOTIC__COCO_train2014_000000342969.jpg', 'EMOTIC__COCO_train2014_000000344031.jpg', 'EMOTIC__COCO_train2014_000000347133.jpg', 'EMOTIC__COCO_train2014_000000349698.jpg', 'EMOTIC__COCO_train2014_000000351610.jpg', 'EMOTIC__COCO_train2014_000000353483.jpg', 'EMOTIC__COCO_train2014_000000355425.jpg', 'EMOTIC__COCO_train2014_000000360017.jpg', 'EMOTIC__COCO_train2014_000000361190.jpg', 'EMOTIC__COCO_train2014_000000362658.jpg', 'EMOTIC__bj6qnim3c43cj2j7n3.jpg', 'EMOTIC__COCO_train2014_000000051735.jpg', 'VOC2012__2007_000063.jpg', 'VOC2012__2007_000068.jpg', 'VOC2012__2007_000793.jpg', 'VOC2012__2007_001487.jpg', 'VOC2012__2007_001583.jpg', 'VOC2012__2007_001704.jpg', 'VOC2012__2007_002565.jpg', 'VOC2012__2007_002643.jpg', 'VOC2012__2007_003104.jpg', 'VOC2012__2008_005594.jpg', 'VOC2012__2008_005600.jpg', 'VOC2012__2008_005607.jpg', 'VOC2012__2008_005615.jpg'])


gr=pd.read_csv('golden_results.csv')
gs=gr.values[:,29]
gsi=np.array([np.fromstring(gs[i],dtype='int',sep=',') for i in range(18)])
gsm=np.mean(gsi, axis=0)
gss=np.std(gsi, axis=0)
ind=np.argsort(gss)
gsm=gsm[ind]
gss=gss[ind]
golden_used = golden_names[ind][:30]
gsm=gsm[:30]
gss=gss[:30]

gind=[np.argwhere(names==golden_used[i])[0,0] for i in range(len(golden_used))]

names_ind = np.delete(np.arange((names.shape[0])),gind)
names=names[names_ind]

num_images=names.shape[0]
#hit_names=np.zeros((3200,N),dtype='str')
hit_names=[]
hit_list_ids=[]
ids=np.arange(num_images)
np.savetxt('counts.csv',np.r_[count], '%d')

#loop, run each hit
print('initial ids: ',ids)
#for count in range(3200):
while(count[0]<3200):
    #if(count*N<num_images+1):
        count=np.genfromtxt('counts.csv','int')*1
        count[0]+=1
        count[1]+=1
        gic = np.random.choice(golden_used,6)
        gic_ind = np.random.choice(np.arange(10),6)
        gic_ind[4:]+=35
        gsm_c= np.array([gsm[golden_used==gic[i]][0] for i in range(len(gic))])
        gss_c= np.array([gss[golden_used==gic[i]][0] for i in range(len(gic))])

        if count[1]*N+N > num_images:
            print('got to end, count*N is %d shuffling:'%count[1])
            count[1]=0
            shuffle(ids)
            print(ids)
        hit_ids=ids[(count[1]*N)%num_images:(count[1]*N)%num_images+N]
        hit_list_ids.append(hit_ids)

        names_c=names[hit_ids]
        for i in range(len(gic)):
            names_c=np.insert(names_c,gic_ind[i],gic[i])
        #hit_names.append(names[hit_ids])
        #hit_names.append(np.r_[ gic[0:3], names[hit_ids],golden_used[3:]])
        hit_names.append(np.r_[names_c,gic_ind,(gsm_c-gss_c).round().astype('int'), (gsm_c+gss_c).round().astype('int')])

        np.savetxt('counts.csv',count, '%d')
    #else:
        #break
hit_names=np.array(hit_names)
hit_list_ids=np.array(hit_list_ids)
np.savetxt('hit_list.csv',hit_names,delimiter=',',fmt='%s')
#urls = np.zeros(names.shape,dtype='str')
#for i in range(len(urls)):
#    
#    print(urls[i])
#    utils.download('https://snako.s3.us-east-2.amazonaws.com/'+names[i], path='8k_images/'+names[i])


