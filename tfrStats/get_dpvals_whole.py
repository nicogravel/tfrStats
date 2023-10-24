import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


# function to get p-values for whole distribution
def get_dpvals_whole(tfr_emp, tfr_null,fband, pk, sigma):

    """
    Get p-values from min-max null distribution

    This functions pools the averaged null distribution values and computes 
    the p-values for each frequency and time bin using the empirical
    cumulative distribution method.
    
    
    .. todo::  
        * Merge this function with get_pvals_whole.

    
    :param float tfr: empirical time frequency representation (i.e. 30, 12, 16, 113 ).
    :param float null_tfr: nul time frequency representation (i.e. 30, 12, 16, 113 ).

    :return: statistical map of p-values for each frequency-time or space-time bin.
    :rtype: float
 
    @author: Nicolas Gravel, 19.09.2023  
    
    """

    #print(tfr_emp.shape, tfr_null.shape)    
    # pool permutations accordingly
    # n_perm = tfr_null.shape[0]
    tfr = np.nanmean(tfr_emp,axis=1)
    #print('tfr after averaging :', tfr.shape)      # frequency x time
    nullDist = tfr_null #np.nanmean(np.nanmean(tfr_null,axis=0),axis=1) # average conditions and sites
    #print('null after averaging :',nullDist.shape) # permutations x frequency x min/max
    stats = np.zeros((tfr_emp.shape[0],tfr_emp.shape[2]))
    #print('stats shape :', stats.shape)

    tps  = [57,113,141,140]
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    t0   = np.searchsorted(time,400,side='left', sorter=None)
    tf   = np.searchsorted(time,1000,side='left', sorter=None)


    for i_site in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            null = nullDist[:,:,t0:tf] # use the both min and max
            obs = np.squeeze(tfr[i_site,i_time])
            ecdf = ECDF(null.flatten())
            p_fwe = ecdf(obs)
            stats[i_site,i_time] = 1.0 - p_fwe
            #stats[i_freq,i_time] = (null >= obs).sum() / n_perm
    return stats

