import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

def get_dpvals_minmax(tfr_emp, tfr_null, tail):

    """
    Get p-values from min-max null distribution

    This functions pools the null distribution values and computes
    the p-values for each frequency and time bin.

    .. todo::
        * Merge this function with get_pvals_whole.


    :param float tfr: empirical time frequency representation (i.e. 30, 12, 16, 113 ).
    :param float null_tfr: null time frequency representation (i.e. 1000, 30, 12, 16, 2 ).

    :return: statistical map of p-values for each frequency-time or space-time bin.
    :rtype: float

    @author: Nicolas Gravel, 19.09.2023

    """
    # pool permutations accordingly
    tfr = np.nanmean(tfr_emp,axis=1)
    stats = np.zeros((tfr_emp.shape[0],tfr_emp.shape[2]))
    for i_site in range(stats.shape[0]):

        if tail == 'single-sided':
            nullDist       = tfr_null[:,:,:,1]
        elif tail == 'two-sided':
            nullDist       = tfr_null[:,:,:,:]

        #nullDist      = np.amax(nullDist, axis=1) # max across frequency bins
        #nullDist      = np.nanmean(nullDist, axis=2) # mean across frequency bins
        #null           = np.amax(nullDist, axis=1) # max across site
        #null           = np.nanmean(nullDist, axis=1) # average across sites (noo!)
        
        null = nullDist
      
        for i_time in range(stats.shape[1]):
            obs = np.squeeze(tfr[i_site,i_time])
            #ecdf = ECDF(null.flatten())
            #p_ = ecdf(obs)
            #stats[i_site,i_time] = 1.0 - p_
            stats[i_site,i_time] = (null >= obs).sum() / tfr_null.shape[0]
    return stats
