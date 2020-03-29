
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


