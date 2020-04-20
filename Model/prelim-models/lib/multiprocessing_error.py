from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import multiprocessing as mp


def gbmerror(parameter):
    name = "maxdepth=" + str(parameter)
    print(name)
    # load data
    X_tr = np.load('X_tr.npy')
    X_ts = np.load('X_ts.npy')
    y_re_tr = np.load('y_re_tr.npy')
    y_re_ts = np.load('y_re_ts.npy')
    y_im_tr = np.load('y_im_tr.npy')
    y_im_ts = np.load('y_im_ts.npy')
    gbm = GradientBoostingRegressor(learning_rate=0.2, n_estimators=100,
                                    max_depth=parameter, min_samples_split=450,
                                    min_samples_leaf=200, max_features=1,
                                    subsample=.5, verbose=10)
    gbm.fit(X_tr, y_re_tr)
    tr = gbm.score(X_tr, y_re_tr)
    ts = gbm.score(X_ts, y_re_ts)
    print(name)
    print('done')
    return tr, ts


if __name__ == '__main__':
    count = mp.cpu_count()
    start = 1
    stop = start+count
    p = mp.Pool(count)
    scores = p.map(gbmerror, range(start, stop))
    scores = np.array(scores)
    np.save('scores', scores)
    p.close()
    p.join()
    print('all done')
