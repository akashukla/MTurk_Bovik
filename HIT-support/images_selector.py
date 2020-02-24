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
golden_names = np.array(['AVA__20278.jpg', 'EMOTIC__4wl5mafxcb0nacmnzr.jpg', 'EMOTIC__COCO_train2014_000000571644.jpg', 'EMOTIC__COCO_val2014_000000031983.jpg', 'EMOTIC__COCO_val2014_000000045685.jpg', 'EMOTIC__COCO_val2014_000000147471.jpg', 'EMOTIC__COCO_val2014_000000464134.jpg', 'EMOTIC__COCO_val2014_000000534041.jpg', 'JPEGImages__2007_003137.jpg', 'JPEGImages__2008_002845.jpg', 'JPEGImages__2008_003688.jpg', 'JPEGImages__2008_003701.jpg', 'JPEGImages__2008_004077.jpg', 'JPEGImages__2008_005266.jpg', 'JPEGImages__2008_006076.jpg', 'JPEGImages__2008_006461.jpg', 'JPEGImages__2008_006554.jpg', 'JPEGImages__2008_006946.jpg', 'JPEGImages__2008_007218.jpg', 'JPEGImages__2008_008544.jpg', 'JPEGImages__2009_000015.jpg', 'JPEGImages__2009_001078.jpg', 'JPEGImages__2009_002872.jpg', 'JPEGImages__2009_002893.jpg', 'JPEGImages__2009_004426.jpg', 'JPEGImages__2010_000088.jpg', 'JPEGImages__2010_002618.jpg', 'JPEGImages__2010_003648.jpg', 'JPEGImages__2010_004257.jpg', 'JPEGImages__2011_000180.jpg', 'JPEGImages__2011_002511.jpg', 'JPEGImages__2011_004079.jpg', 'JPEGImages__2011_006006.jpg', 'JPEGImages__2011_007016.jpg', 'JPEGImages__2011_007090.jpg', 'JPEGImages__2012_000805.jpg', 'JPEGImages__2012_003188.jpg', 'JPEGImages__2012_003896.jpg', 'JPEGImages__2012_004289.jpg', 'VOC2012__2008_001175.jpg', 'VOC2012__2008_001855.jpg', 'VOC2012__2008_002589.jpg', 'VOC2012__2008_003036.jpg', 'VOC2012__2008_004536.jpg', 'VOC2012__2008_008339.jpg', 'VOC2012__2008_008503.jpg', 'VOC2012__2009_000151.jpg', 'VOC2012__2009_000659.jpg', 'VOC2012__2009_000802.jpg', 'VOC2012__2009_001550.jpg', 'VOC2012__2009_002175.jpg', 'VOC2012__2009_002794.jpg', 'VOC2012__2009_003177.jpg', 'VOC2012__2009_003324.jpg', 'VOC2012__2009_004324.jpg', 'VOC2012__2010_000853.jpg', 'VOC2012__2010_005622.jpg', 'VOC2012__2011_002173.jpg', 'VOC2012__2011_002215.jpg', 'VOC2012__2011_002960.jpg', 'VOC2012__2011_003123.jpg', 'VOC2012__2011_003969.jpg', 'VOC2012__2011_005123.jpg', 'VOC2012__2012_001367.jpg', 'VOC2012__2012_003683.jpg'])

N = 43
num_golden = 7
num_repeat = 5
count=np.r_[0,0]

# For actual HIT has 8K images
names=np.genfromtxt('./dat/8k_image_names.csv','str', delimiter=',')
names=names[:,0]

gr=pd.read_csv('golden_results.csv')
gs=gr.values[:,29]
#gs=gr.values
# gsi has all data 18xg50 array of scores
gsi=np.array([np.fromstring(gs[i],dtype='int',sep=',') for i in range(18)])
# array of means
gsm=np.mean(gsi, axis=0)
gss=np.std(gsi, axis=0)
ind=np.argsort(gss)
gsm=gsm[ind]
gss=gss[ind]
# Sorted golden
golden_used = golden_names[ind]#[:30]
#gsm=gsm#[:30]
#gss=gss#[:30]
g
# Remove golden images from big array of names
gind=[np.argwhere(names==golden_used[i])[0,0] for i in range(len(golden_used))]
names_ind = np.delete(np.arange((names.shape[0])),gind)
names=names[names_ind]

# Around 8K
num_images=names.shape[0]
#hit_names=np.zeros((3200,N),dtype='str')

# Init stuff for selection
hit_names=[]
hit_list_ids=[]
ids=np.arange(num_images)
np.savetxt('./gen/counts.csv',np.r_[count], '%d')

#loop, run each hit
print('initial ids: ',ids)
#for count in range(3200):

# Each iteration generates one HIT
while(count[0]<3200):
    #if(count*N<num_images+1):
        count=np.genfromtxt('./gen/counts.csv','int')*1
        # Overall counter since we need to see all 8K images 20x each
        count[0]+=1
        # Resets at 8K
        count[1]+=1

        # Choose 5 random golden images 
        gic = np.random.choice(golden_used,num_golden,replace=False)
        # Places to inset the golden images
        gic_ind = np.random.choice(np.arange(10),num_golden,replace=False)
        # Forces last couple to the end
        gic_ind[5:]+=25
        # Ind of initial occurances (params to arange are the windows)
        rep_ind_b=np.random.choice(np.delete(np.arange(10,20),gic_ind[:5]), num_repeat,replace=False)
        # Where second occurances will be 
        rep_ind_e=np.random.choice(np.delete(np.arange(35,50,1),gic_ind[5:]), num_repeat,replace=False)
        # Sorted
        gic_ind = np.sort(gic_ind)
        rep_ind_b=np.sort(rep_ind_b)
        rep_ind_e=np.sort(rep_ind_e)
        print('gic_ind', gic_ind)
        print('rep_ind_b', rep_ind_b)
        print('rep_ind_e', rep_ind_e)
        # Grabs corresponding golden means and stdevs 
        gsm_c= np.array([gsm[golden_used==gic[i]][0] for i in range(len(gic))])
        gss_c= np.array([gss[golden_used==gic[i]][0] for i in range(len(gic))])
       
        # N is number of images per HIT
        # If we have gone through all 8K images 
        if count[1]*N+N > num_images:
            print('got to end, count*N is %d shuffling:'%count[1])
            count[1]=0
            # shuffle ids of the 8K bc we select 8K by ID
            shuffle(ids)
            print(ids)
        # HIT ids for next set
        hit_ids=ids[(count[1]*N)%num_images:(count[1]*N)%num_images+N]
        # now obsolete
        hit_list_ids.append(hit_ids)

        # Local names for this set loop
        names_c=names[hit_ids]
        # Insets golden images
        for i in range(len(gic)):
            names_c=np.insert(names_c,gic_ind[i],gic[i])
        # Inserts second instance of repeats
        for i in range(len(rep_ind_b)):
            # rep_ind_b = first ind of repeated
            toinsert=names_c[rep_ind_b[i]]
            #names_c=np.insert(names_c,rep_ind_b[i],toinsert)
            names_c=np.insert(names_c,rep_ind_e[i],toinsert)
        #hit_names.append(names[hit_ids])
        #hit_names.append(np.r_[ gic[0:3], names[hit_ids],golden_used[3:]])
        #hit_names.append(np.r_[names_c,gic_ind,(gsm_c-gss_c).round().astype('int'), (gsm_c+gss_c).round().astype('int')])
        # Here is what creates the csv file
        hit_names.append(np.r_[N+num_golden+num_repeat, names_c,gic_ind,gsm_c.round().astype('int'), gss_c.round().astype('int'), rep_ind_b,rep_ind_e])
        np.savetxt('./gen/counts.csv',count, '%d')

    #else:
        #break
hit_names=np.array(hit_names)
hit_list_ids=np.array(hit_list_ids)
np.savetxt('./gen/hit_list.csv',hit_names,delimiter=',',fmt='%s')




#urls = np.zeros(names.shape,dtype='str')
#for i in range(len(urls)):
#    
#    print(urls[i])
#    utils.download('https://snako.s3.us-east-2.amazonaws.com/'+names[i], path='8k_images/'+names[i])




def simulate():
    gic = np.random.choice(golden_used,6,replace=False)
    gic_ind = np.random.choice(np.arange(10),6,replace=False)
    gic_ind[4:]+=35
    gsm_c= np.array([gsm[golden_used==gic[i]][0] for i in range(len(gic))])
    gss_c= np.array([gss[golden_used==gic[i]][0] for i in range(len(gic))])
    amounts=np.zeros(18)
    for i in range(18):
        gsi2=gsi[:,gic_ind]
        indw=np.argwhere(np.absolute(gsi2[i]-np.mean(gsi2,axis=0)) > np.std(gsi2,axis=0))
        wrong=np.count_nonzero(indw)
        amounts[i]=(np.absolute(gsi2[i]-np.mean(gsi2,axis=0))/np.std(gsi2,axis=0) - 1)[indw].sum() 
    
        
        #print('i =', i, ', wrong:', wrong)
        #print('amount:',amounts[i])
    
    print('amounts', amounts, 'perfect ones', np.count_nonzero(amounts==0))
    return amounts
def get_people():
    amount_n=np.zeros((18,10))
    for n in range(10):
        amount_n[:,n] = simulate()
    #pavan=amount_n[0,:]    
    #dallas=amount_n[1,:]
    #mom=amount_n[4,:]
    #brain=amount_n[17,:]
    #print('\n pavan :',pavan)
    #print('\n dallas: ',dallas)
    #print('\n mom: ',mom)
    #print('\n brain: ',brain)

    #return pavan,dallas,mom,brain
    return amount_n

#images, image names, # golden images used, # of repeat images, golden image indice, golden image means, golden image stds, repeated image indices
