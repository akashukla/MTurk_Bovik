# coding: utf-8
corrs=np.zeros(1000)
ps=np.zeros(1000)
for n in range(1000):
    rand_test=np.random.choice(np.arange(20,30),5)
    corrs[n], ps[n] = sp.stats.spearmanr(contra_mu, rand_test)
np.count_nonzero(corrs>.75)/corrs.shape[0]
