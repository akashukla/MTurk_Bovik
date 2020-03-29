# coding: utf-8
b1 = pd.read_csv('batch1_results.csv')
b2 = pd.read_csv('batch2_results.csv')
b3 = pd.read_csv('batch3_results.csv')
b1sn = b1.loc[:,'Answer.set_number'][~(b1.loc[:,'Answer.set_number']=='initial')].astype('int64')
b2sn = b2.loc[:,'Answer.set_number'][~(b2.loc[:,'Answer.set_number']=='initial')].astype('int64')
b3sn = b3.loc[:,'Answer.set_number'][~(b3.loc[:,'Answer.set_number']=='initial')].astype('int64')
ball = np.append(b1sn, b2sn)
ball = np.append(ball, b3sn)

hit_data_orig=np.genfromtxt('hit_data_orig.csv',delimiter=',',dtype='str')[:,1:56]

dic={}
for im in images_orig:
    dic[im]=0

for sn in ball:
    hd_sn=hit_data_orig[sn]
    for im_n in hd_sn:
        dic[im_n]+=1
        
