# coding: utf-8
import numpy as np
import numpy.random as npr
from numpy.random import permutation, shuffle
num_per_hit = 50
num_golden = 5
count=np.r_[0,0]
names=np.genfromtxt('8k_image_names.csv','str', delimiter=',')
num_images=names.shape[0]
names=names[:,0]
#hit_names=np.zeros((3200,50),dtype='str')
hit_names=[]
ids=np.arange(num_images)
np.savetxt('counts.csv',np.r_[count], '%d')

#loop, run each hit
print('initial ids: ',ids)
for count in range(3200):
    #if(count*50<num_images+1):
        count=np.genfromtxt('counts.csv','int')*1
        count[0]+=1
        count[1]+=1
        if count[1]*50 > num_images-50:
            print('got to end, count*50 is %d shuffling:'%count[1])
            count[1]=0
            shuffle(ids)
            print(ids)
        hit_ids=ids[(count[1]*50)%num_images:(count[1]*50)%num_images+50]
        hit_names.append(names[hit_ids])
        np.savetxt('counts.csv',count, '%d')
    #else:
        #break
hit_names=np.array(hit_names)
np.savetxt('hit_list.csv',hit_names,delimiter=',',fmt='%s')
  
