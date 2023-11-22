import os
import numpy as np
import scipy.io as sio
#from acme import ParallelMap, esi_cluster_setup, cluster_cleanup  
from tqdm.auto import tqdm

def mvtfr_SLURM(jobs, cond, fband, split):
    """ 
    Running the mvtfr_classifier module in SLURM


    Function to run ACME, an efficient SLURM manager for Python.
    Given the inputs, selects the data:
    conditions x repetitions x channels x frequency x time points
    and computes, for each frequency and channel, the classification 
    accuracy matrix: conditions x conditions x time points.  
    Because SLURM and ACME only accepts integers as inputs, the data 
    selection occurs inside the function. Paths, conditions and 
    frequency bands.
    
    
    .. todo::  
        * Add the ACME free version (using batch SLURM job arrays)

    
    :param int jobs: a 1d numpy array specifying the number of jobs
    :param int cond: condition index(gratings, natural images)
    :param int fband: frequency band index (low, high and higher frequencies)
    :param int split : index to split (even, odd or whole)

    :return: classification accuracy.
    :rtype: numpy array

    @author: Nicolas Gravel, 19.09.2023  
    """

    #n_jobs = 100
    #client = esi_cluster_setup(timeout=360*4, partition="8GBL", n_jobs=n_jobs)


    conds   = ['grat', 'nat']
    fbands  = ['lf','hf','hhf'] 
    methods = ['hanning', 'wavelet','wavelet']; 

    #methods = ['hanning', 'wavelet','hanning']; 
    splits   = ['split1', 'split2','whole']

    for cond in range(2): 
        for fband in range(3): 
            for split in range(3):
                print('block: ', conds[cond])
                print('method: ', methods[fband])
                print('frequency band: ', fbands[fband])
                print('split: ', splits[split])
            
                if fband == 0:
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(0,100), cond=cond, fband=fband, split=split) as pmap:
                        pmap.compute()
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(100,200), cond=cond, fband=fband, split=split) as pmap:
                            pmap.compute()
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(200,228), cond=cond, fband=fband, split=split) as pmap:
                            pmap.compute()
    
                        
                if fband == 1:
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(0,100), cond=cond, fband=fband, split=split) as pmap:
                        pmap.compute()
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(100,192), cond=cond, fband=fband, split=split) as pmap:
                            pmap.compute()
                            
                if fband == 2:
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(0,66), cond=cond, fband=fband, split=split) as pmap:
                        pmap.compute()
                    with ParallelMap(mvtfr_classifier, jobs=np.arange(66,132), cond=cond, fband=fband, split=split) as pmap:
                            pmap.compute()
      
                     
    return 
                                                                                                                                              