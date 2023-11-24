import os
import numpy as np
import random
from tqdm.auto import tqdm
import scipy.stats
import rsatoolbox
from rsatoolbox.rdm.rdms import RDMs
from scipy import stats as st
from tfrStats.load_tfr_rdm import load_tfr_rdm as load_tfr_rdm
import random

random.seed(111)

def mvtfr_reliability(rdms,conf):  
    """   
    Multivariate Time Frequency Representation Reliability (MVTFR-RSA)


    Taking paths and a configuration structure as input and computes, for each frequency bin 
    and time window, the Spearman’s rho correlation between two RDMs (for ven and odd trials).
    Following on Schütt et al, (2023), here we use the Spearman’s rank-correlation with random
    tiebreaking as an evaluation criterion for RSA instead of original Spearman’s rank-correlation,
    as the later is biased  for tied ranks (for details see Schütt et al, (2023)). Optionally, 
    we assess the contribution to the representational structure that emerges in time-frequency
    correlation map, by using a permutation approach. For each frequency bin in a given time 
    point (an index to an epoque), we proceeded as follows: 

        1. Calculate the correlation of the original RDMs A and B.
        2. Perform a permutation on RDM A (random shuffle) and calculate the correlation with RDM B.
            Repeat the other way around.To save time, this step is performed within a time window of 
            interest (here -250 to 1200 ms). See line 200.
        3. Repeat step 2 a certain number of times, e.g., 100 times, to obtain a null distribution 
            of correlation values.
        4. Calculate the p-value based on the percentage of permuted correlation coefficients greater
            than or equal to the  observed correlation.
        
    This results in spectrograms that are based on time-frequency specific image-related information
    and a null distribution that tests for condition-level significance.

    Originally I was sugin scippy's Kendall's tau but now I use:
    https://rsatoolbox.readthedocs.io/en/stable/comparing.html#spearman-s-rho

    

    Args:
        input_path: a string
        condition: an integer 
        fband: frequency band index (low frequencies, high frequencies), integer
        method: method index (hanning, multitaper, wavelet)
        dvar: spectral variable (GPR or spectral power), integer
        split : index to split (even, odd or whole)

    Returns:
        empirical and nulll channel  x time x frequency distributiont (np.array)

    @author: Nicolas Gravel, 19.09.2023  
    """ 
    
    rdm1 = rdms['rdm_split1']
    rdm2 = rdms['rdm_split2']
    rdm  = rdms['rdm_whole']
    print(rdm.shape)

    paths   = conf['paths']
    fband   = conf['fband']
    cond    = conf['cond']
    dvar    = conf['dvar']
    sess    = conf['sess']
    n_perm  = conf['n_perm']

    if sess != 0:
        prefix = str('_sess_' + str(sess)+ '_') 
        decvars = ["spw", "gpr",'spwgpr']
    else:    
        prefix = str('')
        decvars = ["spw", "gpr",'spwgpr']
        
    tps = [57,113,141,140] 
    fps = [19,16,11,1]

    fbands  = ['lf','hf','hhf','mua']
    blocks   = ['grat', 'nat','nat']
    conds   = ['grat', 'bck','obj','bck-obj','grat_lowcon','grat_highcon','bckXobj']
    
    depths = 12
    methods = ['hanning', 'wavelet', 'wavelet','wavelet']

        
    input_path = paths[1]

  


    # Indexing of conditions
    # =============================================================================
    gratings = [i for i in range(30)] # Total channels (sessions x sites = 40 channels)
    cond_idx = np.zeros((6,5)).astype(np.uint) # Index to the 8 sites in the total channels vector
    for n in range(6): 
        C = [x for x in gratings if x%6 == n]
        cond_idx[n,:] = C
    
    #print("Conditions : ", cond_idx)
    #print(cond_idx.shape)

    #print(cond_idx[0,:])

    if cond == 0:
        block = 0
        n_sess = 10
        results_path = input_path + blocks[block] + '/'
        #trialIdx = cond_idx[:,4] # np.arange(30)
        trialIdx = np.arange(30)
        c1 = trialIdx  # Gratings
        c2 = c1        # Gratings   
        M = np.zeros((c1.shape[0],c2.shape[0]))

    # Object and scenes indices were fliped during the creation of: 'ftPool_' + cond + ...  + '.mat' 
    # =============================================================================
    elif cond == 2:
        block = 1
        n_sess = 11
        results_path = input_path + blocks[block] + '/'
        trialIdx = np.arange(36)
        c1 = trialIdx[1::2]   # Scenes
        #c1 = imgs
        c2 = c1
        M = np.zeros((c1.shape[0],c2.shape[0]))

    elif cond == 1:
        block = 1
        n_sess = 11
        results_path = input_path + blocks[block] + '/'
        trialIdx = np.arange(36)
        c1 = trialIdx[::2]  # Objects
        #c1 = imgs
        c2 = c1
        M = np.zeros((c1.shape[0],c2.shape[0]))

    # =============================================================================
    
    elif cond == 3:
        block = 1
        results_path = input_path + blocks[block] + '/'
        trialIdx = np.arange(36)
        #c1 = trialIdx[1::2]   # Scenes
        #c2 = trialIdx[::2]  # Objects
        c1 = trialIdx
        c2 = c1
        M = np.zeros((c1.shape[0],c2.shape[0]))

    elif cond == 4:
        block = 0
        results_path = input_path + blocks[block] + '/'
        #trialIdx = cond_idx[:,4] # np.arange(30)
        trialIdx = np.arange(30)
        c1 = trialIdx[0:18]  # Gratings
        c2 = c1        # Gratings   
        M = np.zeros((c1.shape[0],c2.shape[0]))

    elif cond == 5:
        block = 0
        results_path = input_path + blocks[block] + '/'
        #trialIdx = cond_idx[:,4] # np.arange(30)
        trialIdx = np.arange(30)
        c1 = trialIdx[19:30]  # Gratings
        c2 = c1        # Gratings  
        M = np.zeros((c1.shape[0],c2.shape[0]))

                    
    # Size of RDM vector to be correlated  
    idx = np.tril_indices(M.shape[0], -1)
                
   
    if fband == 3:
        rsa  = np.zeros((depths,tps[fband]))
        rsa_null  = np.zeros((n_perm*2,depths,tps[fband]))
    else:
        rsa  = np.zeros((depths,fps[fband],tps[fband]))    
        rsa_null  = np.zeros((n_perm*2,depths,fps[fband],tps[fband]))    
       
    for ch in tqdm(range(depths), desc=str(cond), position=0): 

        # Load MUA
        if fband == 3:
            # Load MUA
            X_t = np.load(os.path.join(results_path  
                             + 'unsDec_ch' + str(ch+1) 
                             + '_mua_resampled' 
                             + '_split1'
                             + '.npy'))
            
            Y_t = np.load(os.path.join(results_path  
                             + 'unsDec_ch' + str(ch+1) 
                             + '_mua_resampled' 
                             + '_split2'
                             + '.npy'))
             
            time_tfr = np.linspace(start = -800, stop = 2000, num = 140)
            time_mua = np.linspace(start = -800, stop = 2000, num = 560)             
                        
            for tp in range(len(time_tfr)-1):  
                
                t0 = time_tfr[tp]
                tf = time_tfr[tp+1]

                tt = time_tfr[tp]
                tt0 = np.searchsorted(time_tfr,-200,side='left', sorter=None)
                ttf = np.searchsorted(time_tfr,1200,side='left', sorter=None)
                
                # "zero trail"
                if tt < tt0 or tt >= ttf:
                    rsa[0,ch,tp] = 0
                    rsa_null[:,ch,tp] = np.tile(0,n_perm*2)
            
                # permute
                if tt >= tt0 or tt <ttf:

                    t0_mua = np.searchsorted(time_mua,t0,side='left', sorter=None)
                    td_mua = np.searchsorted(time_mua,tf,side='left', sorter=None)

                    XX = np.mean(X_t[:,:,t0_mua:td_mua],axis=2)                    
                    YY = np.mean(Y_t[:,:,t0_mua:td_mua],axis=2)


                    X = XX[np.ix_(c1,c2)]
                    X = X[~np.eye(X.shape[0],dtype=bool)].reshape(X.shape[0],-1)
                    Y = YY[np.ix_(c1,c2)]
                    Y = Y[~np.eye(Y.shape[0],dtype=bool)].reshape(Y.shape[0],-1)
                
                    rdm_1 = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(X[idx]))
                    rdm_2 = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(Y[idx]))
                    
                    rsa[ch,tp] = rsatoolbox.rdm.compare_rho_a(rdm_1,rdm_2)[0]

                    
                    for rep in range(n_perm): 


                        c_rnd = random.sample(list(c1), len(c1))

                        X_null  = XX[np.ix_(c_rnd,c_rnd)]
                        X_null  = X_null[~np.eye(X_null.shape[0],dtype=bool)].reshape(X_null.shape[0],-1) 
                        Y_null  = YY[np.ix_(c_rnd,c_rnd)] 
                        Y_null  = Y_null[~np.eye(Y_null.shape[0],dtype=bool)].reshape(Y_null.shape[0],-1) 
                                        
                        rdm_1_null = RDMs(dissimilarity_measure = 'classification accuracy',dissimilarities = np.array(X_null[idx]))
                        rdm_2_null = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(Y_null[idx]))

                        rsa_null[rep,ch,tp]    = rsatoolbox.rdm.compare_rho_a(rdm_2,rdm_1_null)[0]            
                        rsa_null[rep+n_perm,ch,tp] = rsatoolbox.rdm.compare_rho_a(rdm_1,rdm_2_null)[0]  
                    
           
        else:

            for fr in range(fps[fband]):

                #(12, 16, 30, 30, 113)

                X_t = rdm1[ch,fr,:,:,:]               
                Y_t = rdm2[ch,fr,:,:,:]

                for tp in range(tps[fband]):

                    time = np.linspace(start = -800, stop = 2000, num = tps[fband])
                    tt = time[tp]
                    tt0 = np.searchsorted(time,-150,side='left', sorter=None)
                    ttf = np.searchsorted(time,1200,side='left', sorter=None)
                    
                    # "zero trail"
                    if tt < tt0 or tt>=ttf:
                        rsa[ch,fr,tp] = 0
                        rsa_null[:,ch,fr,tp] = np.tile(0,n_perm*2)
                
                    # permute
                    if tt >= tt0 or tt <ttf:
                         
                        if cond == -5 or cond == -4 or cond == -3 or cond == -2 or cond == -1:
                            XX = np.squeeze(X_t[:,:,tp])*mask+mask_2
                            YY = np.squeeze(Y_t[:,:,tp])
                        else:
                            XX = np.squeeze(X_t[:,:,tp])
                            YY = np.squeeze(Y_t[:,:,tp])

                        X = XX[np.ix_(c1,c2)]
                        X = X[~np.eye(X.shape[0],dtype=bool)].reshape(X.shape[0],-1)
                        Y = YY[np.ix_(c1,c2)]
                        Y = Y[~np.eye(Y.shape[0],dtype=bool)].reshape(Y.shape[0],-1)


                        rdm_1 = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(X[idx]))
                        rdm_2 = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(Y[idx]))
                        
                        rsa[ch,fr,tp] = rsatoolbox.rdm.compare_rho_a(rdm_1,rdm_2)[0]
                        

                        for rep in range(n_perm):            
                            
                            c_rnd = random.sample(list(c1), len(c1))

                            X_null  = XX[np.ix_(c_rnd,c_rnd)]
                            X_null  = X_null[~np.eye(X_null.shape[0],dtype=bool)].reshape(X_null.shape[0],-1) 
                            Y_null  = YY[np.ix_(c_rnd,c_rnd)] 
                            Y_null  = Y_null[~np.eye(Y_null.shape[0],dtype=bool)].reshape(Y_null.shape[0],-1) 
                                            
                            rdm_1_null = RDMs(dissimilarity_measure = 'classification accuracy',dissimilarities = np.array(X_null[idx]))
                            rdm_2_null = RDMs(dissimilarity_measure = 'classification accuracy', dissimilarities = np.array(Y_null[idx]))

                            rsa_null[rep,ch,fr,tp]    = rsatoolbox.rdm.compare_rho_a(rdm_1,rdm_2_null)[0]            
                            rsa_null[rep+n_perm,ch,fr,tp] = rsatoolbox.rdm.compare_rho_a(rdm_2,rdm_1_null)[0]        
                       
    return rsa, rsa_null