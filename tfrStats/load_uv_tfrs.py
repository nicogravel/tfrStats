import numpy as np


def load_uv_tfrs(input_path, svar, cond, fband, obs):

    """
    Load empirical and null TFRs

    This functions loads the results of tfr_spw_stats_minmax or load_uv_tfrs
    

    .. todo::  
        * Handle parameters with dictionary. 


    :param string input_path: path to the .npz file.
    :param in condition: condition index (i.e. 0, 1, 2, 3).
    :param int svar: spectral power or GPR (not implemented here).
    :param int fband: frequency band index (i.e. low, high, higher).
    :param int obs: [nullType, percentile], two integeres: 0 for min-max, 1 for whole, 0-100 percentile

    
    :return: empirical time frequency representation n_conds x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113).
    :return: null time frequency representation (i.e. 30, 12, 16, 113  or 1000, 30, 12, 16, 2).
    :rtype: float
 
    @author: Nicolas Gravel, 19.09.2023 
    """



    blocks  = ['grat', 'nat']
    svars   = ['spw', 'gpr']
    fbands  = ['low','high','higher']
    results   = ['_100', '_1000_minmax','_30_roll', '_100_minmax_roll']

    svar = 0



    # Condition index
    if cond == 0:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
    if cond == 1:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
    if cond == 2:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')
    if cond == 3:
        fname = str(input_path +'uvtfr_stats_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + results[obs] + '.npz')




    print(fname)
    npz = np.load(fname)

    # Empirical TFR
    tfr_emp  = npz['arr_0']
    print('tfr emp  : ', tfr_emp.shape)

    # Null TFR
    tfr_null = npz['arr_1']
    print('tfr null   ', tfr_null.shape)


    return tfr_emp, tfr_null
