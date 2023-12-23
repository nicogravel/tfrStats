import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

def get_pvals_minmax(tfr_emp, tfr_null, tail):

    """
    Get p-values from min-max null distribution

    This functions pools the permuted min-max values and computes
    the p-values for each frequency and time bin using the empirical
    cumulative distribution method.


    .. todo::
        * Merge this function with get_pvals_whole.


    :param float tfr: empirical time frequency representation (i.e. 30, 12, 16, 113 ).
    :param float null_tfr: nul time frequency representation (i.e. 1000, 30, 12, 16, 2 ).

    :return: statistical map of p-values for each frequency-time or space-time bin.
    :rtype: float

    @author: Nicolas Gravel, 19.09.2023
    """

    #n_perm = tfr_null.shape[0]
    tfr = np.nanmean(np.nanmean(tfr_emp,axis=0),axis=0) # average conditions and sites
    
    if tail == 'two-sided':
        nullDist  = tfr_null[:,:,:,:,:]  # use the both min and max
    elif tail == 'single-sided':
        nullDist   = tfr_null[:,:,:,:,1]  # use the max
    
    #nullDist     = np.nanmean(nullDist,axis=1) # average conditions
    #nullDist     = np.nanmean(nullDist,axis=1) # max across sites
    null         = nullDist
    stats = np.zeros((tfr_emp.shape[2],tfr_emp.shape[3]))

    for i_freq in range(stats.shape[0]):

        for i_time in range(stats.shape[1]):
            obs = np.squeeze(tfr[i_freq,i_time])
            #ecdf = ECDF(null.flatten())
            #p_ = ecdf(obs)
            #stats[i_freq,i_time] = 1.0 - p_
            stats[i_freq,i_time] = (null >= obs).sum() / tfr_null.shape[0]
    return stats
