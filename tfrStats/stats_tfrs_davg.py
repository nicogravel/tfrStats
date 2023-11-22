import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial.distance import cdist

# dimensions : (16, 113) (200, 12, 16, 113) (200, 12, 16, 113)

def stats_tfrs_davg(tfr_null, tfr_emp, correction):
    """
    Compute p-values obtained from 2-fold permutation RSA tests

    This functions computes p-values obtained from TFR-RSA.

    .. todo::
        * Adapt the function to work with 3D arrays (i.e. 2D space-time-frequency maps).
        * Add the possibility to use a different distance metric.
        * Add the possibility to use a different cluster correction method.

    :param float stat: un-corrected p-values for each frequency (and/or time) bin.
    :param float alpha: statistical threshold (e.g. 0.05).

    :return: corrected p-values for each frequency-time or space-time bin.
    :rtype: float

    @author: Nicolas Gravel, 19.09.2023
    """

    n_perm   = tfr_null.shape[0] # (12, 16, 113) (20, 12, 16, 113)
    tfr      =  tfr_emp #np.nanmean(tfr_emp,axis=1) # average sites
    nullDist  = tfr_null
    
    print('null:',nullDist.shape)

    if correction == "space":
        nullDist  = np.amax(nullDist,axis=1) # max across sites
    elif correction == "frequency":
        nullDist  = np.amax(nullDist,axis=2) # max across freqs
    elif correction == "space-frequency":
        nullDist  = np.amax(nullDist,axis=(1,2)) # max across freqs

    print(nullDist.shape)
    print('dimensions :', tfr.shape, tfr_null.shape, nullDist.shape)
    stats  = np.zeros((tfr.shape[0],tfr.shape[1]))

    time =  np.linspace(start = -800, stop = 2000, num = tfr_null.shape[2])
    #t0  = np.searchsorted(time,400,side='left', sorter=None)
    #tf  = np.searchsorted(time,1000,side='left', sorter=None)

    for i_site in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            if correction == "space":
                null = nullDist[:,:,i_time]
            elif correction == "frequency":
                null = nullDist[:,:,i_time]
            elif correction == "space-frequency":
                null = nullDist[:,i_time]
            obs = np.squeeze(tfr[i_site,i_time])
            ecdf = ECDF(null.flatten())
            p_fwe = ecdf(obs)
            stats[i_site,i_time] = 1.0 - p_fwe
            #stats[i_freq,i_time] = (null >= obs).sum() / n_perm
    return stats
