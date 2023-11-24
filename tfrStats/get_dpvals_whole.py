import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF



def get_dpvals_whole(tfr_emp, tfr_null,fband):

    """
    Get p-values from min-max null distribution

    This functions pools the null distribution values and computes
    the p-values for each frequency and time bin.

    .. todo::
        * Merge this function with get_pvals_whole.


    :param float tfr: empirical time frequency representation (i.e. 30, 12, 16, 113 ).
    :param float null_tfr: nul time frequency representation (i.e. 30, 12, 16, 113 ).

    :return: statistical map of p-values for each frequency-time or space-time bin.
    :rtype: float

    @author: Nicolas Gravel, 19.09.2023

    """

    #n_perm = tfr_null.shape[0]
    tfr = np.nanmean(tfr_emp,axis=1)
    nullDist = tfr_null
    stats = np.zeros((tfr_emp.shape[0],tfr_emp.shape[2]))

    tps  = [57,113,141,140]
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    t0   = np.searchsorted(time,400,side='left', sorter=None)
    tf   = np.searchsorted(time,1000,side='left', sorter=None)


    for i_site in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            null = nullDist[:,:,t0:tf]
            #null = nullDist[:,:,:]
            obs = np.squeeze(tfr[i_site,i_time])
            ecdf = ECDF(null.flatten())
            p_ = ecdf(obs)
            stats[i_site,i_time] = 1.0 - p_
            #stats[i_freq,i_time] = (null >= obs).sum() / n_perm
    return stats
