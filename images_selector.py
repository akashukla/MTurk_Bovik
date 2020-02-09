# coding: utf-8
import gluoncv
from gluoncv import utils
import numpy as np
import numpy.random as npr
from numpy.random import permutation, shuffle
import os, csv
import numpy as np
from array import *

N = 45
num_golden = 5
count=np.r_[0,0]
names=np.genfromtxt('8k_image_names.csv','str', delimiter=',')
golden_names = np.array(['EMOTIC__COCO_train2014_000000208055.jpg', 'EMOTIC__COCO_train2014_000000211272.jpg', 'EMOTIC__COCO_train2014_000000211486.jpg', 'EMOTIC__COCO_train2014_000000212083.jpg', 'EMOTIC__COCO_train2014_000000221881.jpg', 'EMOTIC__COCO_train2014_000000222140.jpg', 'EMOTIC__COCO_train2014_000000223888.jpg', 'EMOTIC__COCO_train2014_000000226597.jpg', 'EMOTIC__COCO_train2014_000000229188.jpg', 'EMOTIC__COCO_train2014_000000230433.jpg', 'EMOTIC__COCO_train2014_000000239307.jpg', 'EMOTIC__COCO_train2014_000000241421.jpg', 'EMOTIC__COCO_train2014_000000250357.jpg', 'EMOTIC__COCO_train2014_000000257297.jpg', 'EMOTIC__COCO_train2014_000000265100.jpg', 'EMOTIC__COCO_train2014_000000266479.jpg', 'EMOTIC__COCO_train2014_000000268209.jpg', 'EMOTIC__COCO_train2014_000000276128.jpg', 'EMOTIC__COCO_train2014_000000277122.jpg', 'EMOTIC__COCO_train2014_000000297359.jpg', 'EMOTIC__COCO_train2014_000000300598.jpg', 'EMOTIC__COCO_train2014_000000301855.jpg', 'EMOTIC__COCO_train2014_000000302102.jpg', 'EMOTIC__COCO_train2014_000000304548.jpg', 'EMOTIC__COCO_train2014_000000305105.jpg', 'EMOTIC__COCO_train2014_000000307894.jpg', 'EMOTIC__COCO_train2014_000000308353.jpg', 'EMOTIC__COCO_train2014_000000311706.jpg', 'EMOTIC__COCO_train2014_000000318496.jpg', 'EMOTIC__COCO_train2014_000000319690.jpg', 'EMOTIC__COCO_train2014_000000319905.jpg', 'EMOTIC__COCO_train2014_000000322212.jpg', 'EMOTIC__COCO_train2014_000000325981.jpg', 'EMOTIC__COCO_train2014_000000326504.jpg', 'EMOTIC__COCO_train2014_000000327810.jpg', 'EMOTIC__COCO_train2014_000000329587.jpg', 'EMOTIC__COCO_train2014_000000329942.jpg', 'EMOTIC__COCO_train2014_000000334338.jpg', 'EMOTIC__COCO_train2014_000000341623.jpg', 'EMOTIC__COCO_train2014_000000341905.jpg', 'EMOTIC__COCO_train2014_000000342969.jpg', 'EMOTIC__COCO_train2014_000000344031.jpg', 'EMOTIC__COCO_train2014_000000347133.jpg', 'EMOTIC__COCO_train2014_000000349698.jpg', 'EMOTIC__COCO_train2014_000000351610.jpg', 'EMOTIC__COCO_train2014_000000353483.jpg', 'EMOTIC__COCO_train2014_000000355425.jpg', 'EMOTIC__COCO_train2014_000000360017.jpg', 'EMOTIC__COCO_train2014_000000361190.jpg', 'EMOTIC__COCO_train2014_000000362658.jpg', 'EMOTIC__bj6qnim3c43cj2j7n3.jpg', 'EMOTIC__COCO_train2014_000000051735.jpg', 'VOC2012__2007_000063.jpg', 'VOC2012__2007_000068.jpg', 'VOC2012__2007_000793.jpg', 'VOC2012__2007_001487.jpg', 'VOC2012__2007_001583.jpg', 'VOC2012__2007_001704.jpg', 'VOC2012__2007_002565.jpg', 'VOC2012__2007_002643.jpg', 'VOC2012__2007_003104.jpg', 'VOC2012__2008_005594.jpg', 'VOC2012__2008_005600.jpg', 'VOC2012__2008_005607.jpg', 'VOC2012__2008_005615.jpg'])


with open('golden_results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    results = [0 for x in range(19)]
    
    for row in csv_reader:
        if line_count > 0:
            result_s = str(row[-3:-2])
            result_s = result_s[2:-3]
            result = [int(s) for s in result_s.split(',')]
            results[line_count - 1] = np.array(result)
            
        
        line_count += 1
    results = np.array(results[:-1])
    golden_mean = np.zeros(65)
    golden_var = np.zeros(65)
    for col in range(65):
        golden_mean[col] = np.mean(results[:,col])
        golden_var[col] = np.var(results[:,col])
        
    print(golden_mean)
    print(golden_var)
    
    idx = np.argpartition(golden_var, 50)
    idx = idx[0:50]
    print(idx)
    print(golden_var[idx[:50]])

golden_used=golden_names[idx[:5]]
num_images=names.shape[0]
names=names[:,0]
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
        if count[1]*N+N > num_images:
            print('got to end, count*N is %d shuffling:'%count[1])
            count[1]=0
            shuffle(ids)
            print(ids)
        hit_ids=ids[(count[1]*N)%num_images:(count[1]*N)%num_images+N]
        hit_list_ids.append(hit_ids)
        #hit_names.append(names[hit_ids])
        hit_names.append(np.r_[ golden_used[0:3], names[hit_ids],golden_used[3:]])
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


