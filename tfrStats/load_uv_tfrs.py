import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors


# function to load the .npz file produced by tfr_spw_stats_minmax or tfr_spw_stats
def load_uv_tfrs(input_path, svar, cond, fband, obs):

    """
    Load empirical and null TFRs

    This functions loads the results of tfr_spw_stats_minmax or load_uv_tfrs
    
    Args:
        input path: string 
        condition: an integer
        svar: spectral power or GPR (not implemented here), integer
        fband: frequency band index (i.e. low, high, higher), integer
        obs: [nullType, percentile], two integeres: 0 for min-max, 1 for whole, 0-100 percentile

    Returns and saves:
        
        In case of min-max:
        empirical tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )
        null tfr: n_perm x n_conds x n_sites x frequencies x min/max (i.e. 1000, 30, 12, 16, 2 )

        In case of whole-null
        empirical tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )
        null tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )

    @author: Nicolás Gravel, 19.09.2023  
    
  
    
    """



    blocks  = ['grat', 'nat','nat','nat','nat']
    svars   = ['spw', 'gpr']
    fbands  = ['low','high','higher']
    results   = ['_100', '_1000_minmax']

    svar = 0

    gratings = [i for i in range(30)] # Total channels (sessions x sites = 40 channels)
    cond_idx = np.zeros((6,5)).astype(np.uint) # Index to the 8 sites in the total channels vector
    for n in range(6):
        C = [x for x in gratings if x%6 == n]
        cond_idx[n,:] = C


    # Condition index
    if cond == 0:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
        trialIdx = np.arange(30)    # Gratings
    if cond == 1:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
        trialIdx = np.arange(36)    # Objects
        trialIdx = trialIdx[::2]
    if cond == 2:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
        trialIdx = np.arange(36)    # Scenes
        trialIdx = trialIdx[1::2]
    if cond == 3:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
        trialIdx = np.arange(36)    # Scenes + objects




    print(fname)
    npz = np.load(fname)

    # Empirical TFR
    tfr_emp  = npz['arr_0']
    print('tfr emp  : ', tfr_emp.shape)

    # Null TFR
    tfr_null = npz['arr_1']
    print('tfr null   ', tfr_null.shape)


    return tfr_emp, tfr_null
