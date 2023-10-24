import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial.distance import cdist

def cluster_correction(stats, cluster_size, alpha):

    """
    Correct p-values obtained from min-max or whole null distributions
    using nearest neighbours cluster correction

    This functions computes Minkowski unweighted distances between all time-frequency bins,
    then,  based on a distance threshold (defined by cluster size) pool the p-vals from the 
    nearest neighbours, tests that all the pvalues are below alpha (center bin and and its neighbours),
    and if true, average the p-values within the cluster and assign the resulting average as
    corrected p-value to the time-frequency bin. 

    .. todo::  
        * Adapt the function to work with 3D arrays (i.e. 2D space-time-frequency maps).
        * Add the possibility to use a different distance metric.
        * Add the possibility to use a different cluster correction method (e.g. using permutation-based percentiles as input).
    
    :param float stat: un-corrected p-values for each frequency (and/or time) bin.
    :param float alpha: statistical threshold (e.g. 0.05).

    :return: corrected p-values for each frequency-time or space-time bin.
    :rtype: float
    
    @author: Nicolas Gravel, 19.09.2023  
    """ 
        
    ## cluster correction

    clusters =  np.ones((stats.shape))
    x_idx, y_idx  = np.where(clusters)
    idx = np.vstack((x_idx, y_idx)).T
    dists = cdist(idx,idx, 'minkowski', p=2.0)
    pvals = stats.flatten()
    pval_corr = np.ones((stats.shape))  
    for i_x in range(stats.shape[0]):
        for i_y in range(stats.shape[1]):
            p   = stats[i_x, i_y]
            if p <= alpha:
                index = np.where((x_idx == i_x) & (y_idx == i_y))
                d = np.squeeze(dists[index,:])
                neighbours = d <= cluster_size
                pval_nn= pvals[neighbours]
                pp  = pval_nn <= alpha
                #if np.count_nonzero(pp) >= np.count_nonzero(neighbours):
                if np.count_nonzero(pp) >= cluster_size:
                    pval_corr[i_x , i_y] = np.mean(np.append(pval_nn,p))

    return pval_corr
