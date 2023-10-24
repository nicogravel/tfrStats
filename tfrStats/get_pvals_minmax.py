import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors

# function to get p-values
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

    # pool permutations accordingly
    n_perm = tfr_null.shape[0]*tfr_null.shape[2] # permutations x sites
    tfr = np.nanmean(np.nanmean(tfr_emp,axis=0),axis=0) # average conditions and sites
    print(tfr.shape)      # frequency x time
    nullDist = np.nanmean(np.nanmean(tfr_null,axis=1),axis=1) # average conditions and sites
    print('hola:',nullDist.shape) # permutations x frequency x min/max
    stats = np.zeros((tfr_emp.shape[2],tfr_emp.shape[3]))
    print(stats.shape)
    for i_freq in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            if tail == 'two-sided':
                null = nullDist[:,:,:] # use the both min and max
            elif tail == 'single-sided':
                null  = nullDist[:,:,1] # use the max
            obs = np.squeeze(tfr[i_freq,i_time])
            ecdf = ECDF(null.flatten())
            p_fwe = ecdf(obs)
            stats[i_freq,i_time] = 1.0 - p_fwe
            #stats[i_freq,i_time] = (null >= obs).sum() / n_perm
    return stats

