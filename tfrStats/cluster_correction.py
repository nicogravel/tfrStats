import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors

# Correct the p-values for multiple comparisons using cluster correction
def cluster_correction(stats, tfr_null, cluster_size, alpha):

    """
    Get p-values from min-max null distribution

    This functions computes distances between all time-frequency bins, pool p-vals 
    of nearest neighbours based on distance threshold (defined by cluster size), tests
    that a time-frequency is below alpha and its neighbours p-vals below alpha, and if
    true, average the p-values within the cluster and assign the resulting average as
    corrected p-value to the time-frequency bin. 

    
    Args:
        stats: un-corrected p-values for each frequency and time bin

    Returns:

        stats: corrected p-values for each frequency and time bin

    @author: Nicolás Gravel, 19.09.2023  
    
    https://nicogravel.github.io/
    
    """
        
    ## cluster correction
    from scipy.spatial.distance import cdist
    clusters =  np.ones((stats.shape))
    x_indexes, y_indexes  = np.where(clusters)
    idx = np.vstack((x_indexes, y_indexes )).T
    dists = cdist(idx,idx)
    pvals = stats.flatten()
    pval_corr = np.ones((stats.shape))  
    for i_freq in range(stats.shape[0]):
        for i_time in range(stats.shape[1]):
            p   = stats[i_freq, i_time]
            if p <= alpha:
                index = np.where((x_indexes == i_freq) & (y_indexes == i_time))
                d = np.squeeze(dists[index,:])
                neighbours = d <= cluster_size
                pvals[neighbours]
                pval_nn= pvals[neighbours]
                p  = pval_nn <= alpha
                if np.count_nonzero(p) > cluster_size:
                    pval_corr[i_freq, i_time] = np.mean(pval_nn)

    return pval_corr
    
