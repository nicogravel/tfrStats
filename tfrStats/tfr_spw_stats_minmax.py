import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors


def tfr_spw_stats_minmax(paths, cond, svar, fband, n_perm):
    

    """
    Permutation based TFR statistical asessement based on min-max

    Function to compute the truncated min-max distribution keeping the permutations 
    for each condition and recording site. It captures the variations at the extreme 
    of the null ditribution. In the min-max approach the minimum and maximum values 
    at each permutations are used. 
    
    .. todo::  
        *  Implement onset shifting to account for whole triakl (in the current example we pool values from the 400-1000 ms time window).
        *  Implement compatibilityu with Syncopy (for now it relies on ftPool_... .mat containing the TFRs computed in Fieldtrip).


    :param string input_path: path to the .npz file.
    :param in condition: condition index (i.e. 0, 1, 2, 3).
    :param int svar: spectral power or GPR (not implemented here).
    :param int fband: frequency band index (i.e. low, high, higher).
    :param int obs: [nullType, percentile], two integeres: 0 for min-max, 1 for whole, 0-100 percentile.
    :param int correction: 1 for p-values, 2 for cluster corrected p-values.
    :param int cluster_size: cluster size.
    :param float alpha: alpha.    
    
    :return: empirical time frequency representation n_conds x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113).
    :return: null time frequency representation (i.e. 30, 12, 16, 113  or 1000, 30, 12, 16, 2).
    :rtype: float
 
    @author: Nicolas Gravel, 19.09.2023 
    """

    tps = [57,113,141,140]
    fps = [19,16,11,1]
    fbands  = ['low','high','higher']
    blocks  = ['grat', 'nat']
    svars   = ['spw', 'gpr']
    methods = ['hanning', 'wavelet','wavelet']
    svar = 0
    # Conditions
    if cond == 0:
        block = 0
        n_sess = 10
    # =============================================================================
    else:
        block = 1
        n_sess = 11


    # =============================================================================
    # How the indices are organized within the dataset
    # =============================================================================
    channels = [i for i in range(12*n_sess)] # Total channels
    site_idx = np.zeros((12,n_sess)).astype(np.uint) # Index to sites
    for n in range(12): # for time
            site = [x for x in channels if x%12 == n]
            site_idx[n,:] = site
    print('site indices :')
    print(site_idx)

    if fband == 0:
         bs_t0 = -700
         bs_t1 = -100
    elif fband == 1:
         bs_t0 = -700
         bs_t1 = -100
    elif fband == 2:
         bs_t0 = -700
         bs_t1 = -100

    # =============================================================================
    # Empirical TFR
    # =============================================================================
    fname = str(paths[0]
                + 'ftPool_'
                + blocks[block] + '_'
                + fbands[fband] + '_'
                + methods[fband] + '.mat')
    print(fname)
    mat = sio.loadmat(fname)
    dataPool = mat.get(str('dataLump_' + svars[svar]))



    print(dataPool.shape)
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    b0 = np.searchsorted(time,bs_t0,side='left', sorter=None)
    bf = np.searchsorted(time,bs_t1,side='left', sorter=None)

    tfr_ = np.zeros((dataPool.shape[0],dataPool.shape[1],12,dataPool.shape[3],dataPool.shape[4]))

    for i_cond in range(dataPool.shape[0]):
        for i_rep in range(dataPool.shape[1]):
            for i_depth in range(12):
                for i_freq in range(dataPool.shape[3]):
                    X = dataPool[i_cond,i_rep,site_idx[i_depth,:],i_freq,:]
                    X = np.nanmean(X,axis=0) # average sessions
                    baseline = dataPool[:,:,site_idx[i_depth,:],i_freq,b0:bf]
                    baseline = np.nanmean(baseline,axis=2) # average time
                    X_bs = np.nanmean(baseline.flatten())
                    tfr_[i_cond,i_rep,i_depth,i_freq,:] =  ((X-X_bs)/X_bs)*100

    tfr_[tfr_ == -inf] = np.nan
    tfr_[tfr_ == inf]  = np.nan
    tfr_emp =  np.nanmean(tfr_,axis=1) # repetition average

    # =============================================================================
    # Null TFR
    # =============================================================================
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    b0  = np.searchsorted(time,bs_t0,side='right', sorter=None)
    bf  = np.searchsorted(time,bs_t1,side='right', sorter=None)
    t0  = np.searchsorted(time,400,side='left', sorter=None)
    tf  = np.searchsorted(time,1000,side='left', sorter=None)
    win = time[t0:tf]
    X_h0   = np.zeros((10,dataPool.shape[3],dataPool.shape[4]))
    tfr_null = np.zeros((n_perm,dataPool.shape[0],12,2))

    msg = (str(cond) + ' - ' + str(blocks[block]) + ' - ' + str(fbands[fband]))
    choices = np.random.random(n_perm) > 0.5
    for i_perm in tqdm(range(n_perm),desc=msg, position=0):
        for i_cond in range(dataPool.shape[1]):
            for i_depth in range(12):
                for i_freq in range(dataPool.shape[3]):
                    for i_rep in range(dataPool.shape[1]):
                        if choices[i_perm] == True:
                            X = dataPool[:,:,site_idx[i_depth,:],i_freq,t0:tf]
                            X = np.nanmean(X,axis=3) # average time
                            X = np.nanmean(X.flatten())
                            XX = np.tile(X,[1,win.shape[0]])
                            X_bs = dataPool[i_cond,i_rep,site_idx[i_depth,:],i_freq,b0:bf]
                            XX_bs = np.nanmean(X_bs,axis=0) # average sessions
                            X_h0[i_rep,i_freq,t0:tf] = ((XX_bs-XX)/XX)*100
                        elif choices[i_perm] == False:
                            X = dataPool[i_cond,i_rep,site_idx[i_depth,:],i_freq,:]
                            X = np.nanmean(X,axis=0) # average sessions
                            baseline = dataPool[:,:,site_idx[i_depth,:],i_freq,b0:bf]
                            baseline = np.nanmean(baseline,axis=3) # average time
                            X_bs = np.nanmean(baseline.flatten())
                            XX_bs = np.tile(X_bs,[1,dataPool.shape[4]])
                            X_h0[i_rep,i_freq,:] = ((X-XX_bs)/XX_bs)*100
                X_h0[X_h0 == -inf] = np.nan
                X_h0[X_h0 == inf]  = np.nan
                X = X_h0[:,:,t0:tf] # pool repetitions, frequency bins (all..) and time bins (400-1000ms))
                # save permutation's min-max for each condition and depth
                tfr_null[i_perm,i_cond,i_depth,0] = np.nanmin(X.flatten())
                tfr_null[i_perm,i_cond,i_depth,1] = np.nanmax(X.flatten())


    print(tfr_emp.shape)
    print(tfr_null.shape)

    fname = str(paths[1] + 'uvtfr_stats_' +  fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + '_' + str(n_perm) + '_minmax.npz')
    print(fname)
    np.savez(fname, tfr_emp, tfr_null)

    return tfr_emp, tfr_null
