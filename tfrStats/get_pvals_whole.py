import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


def get_pvals_whole(tfr_emp, tfr_null,fband):

    """
    Get p-values from whole null distribution

    This functions pools the averaged null distribution values and computes
    the p-values for each frequency and time bin using the empirical
    cumulative distribution method.

    .. todo::
        * Adapt the function to work with N-dimensional arrays from different null realizations.
        * Add the option to use the min or max values from the null distribution or the whole distribution.


    :param float tfr: empirical time frequency representation (i.e. 30, 12, 16, 113 ).
    :param float null_tfr: nul time frequency representation (i.e. 30, 12, 16, 113 ).

    :return: statistical map of p-values for each frequency-time or space-time bin.
    :rtype: float

    @author: Nicolas Gravel, 19.09.2023
    """

    #n_perm = tfr_null.shape[0]
    tfr = np.nanmean(np.nanmean(tfr_emp,axis=0),axis=0) # average conditions and sites
    nullDist = np.nanmean(np.nanmean(tfr_null,axis=0),axis=0) # average conditions and sites
    stats = np.zeros((tfr_emp.shape[2],tfr_emp.shape[3]))

    tps = [57,113,141,140]
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    t0  = np.searchsorted(time,400,side='left', sorter=None)
    tf  = np.searchsorted(time,1000,side='left', sorter=None)


    for i_freq in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            null = nullDist[:,t0:tf] # use the both min and max
            obs = np.squeeze(tfr[i_freq,i_time])
            ecdf = ECDF(null.flatten())
            p_ = ecdf(obs)
            stats[i_freq,i_time] = 1.0 - p_
            #stats[i_freq,i_time] = (null >= obs).sum() / n_perm
    return stats
