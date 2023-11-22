import os
import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def mvtfr_classifier(jobs, cond, fband, split):

    """ 
    
    Linear Discmininant Analysis classification 


    Function to decode stimulus conditions in electrophysiological data
    using a Linear Discriminant Analysis classifier from sklearn. 
    At the moment it relies on the python package ACME, an efficient SLURM 
    manager to use in high performance clusters (HPC), to speed up computations.
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

    paths = [
    '/mnt/hpc/projects/MWlamprop/Analysis/02_data/preprocessed/',
    #'/mnt/pns/home/graveln/Documents/MWlamprop/results/spectral_dec/'
    '/mnt/hpc_slurm/projects/MWlamprop/Analysis/02_data/spectral_dec/'        
    ]
    
    #tps = [57,113,187,140]
    #fps = [19,16,25,1]
    tps = [57,113,141,140] # wavelet
    fps = [19,16,11,1]
    
    conds   = ['grat', 'nat']
    fbands  = ['lf','hf','hhf']
    fbands  = ['low','high','higher']
    
    methods = ['hanning', 'wavelet','wavelet']
    #methods = ['hanning', 'wavelet','hanning']
    svars   = ['spw', 'gpr']    
    splits   = ['split1', 'split2','whole']

    data_path = paths[0]
    results_path = paths[1] + conds[cond] + '/'
    fname = str(data_path + 'ftPool_' + conds[cond] + '_' + fbands[fband] + '_' + methods[fband] + '.mat')
    print(fname)
    mat = sio.loadmat(fname)
    #dataPool_spw = mat.get(str('dataLump_' + svars[0]))
    dataPool = mat.get(str('dataLump_' + svars[0]))
    #dataPool_gpr = mat.get(str('dataLump_' + svars[1]))
    
    ### Normalization
    tps = [57,113,187,140]
    fps = [19,16,25,1]
    tps = [57,113,141,140] # wavelet
    fps = [19,16,11,1]
    
    if fband == 0:
         bs_t0 = -700
         bs_t1 = -100
    elif fband == 1:
         bs_t0 = -700
         bs_t1 = -100
    elif fband == 2:
         bs_t0 = -700
         bs_t1 = -100
        
    time =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    b0 = np.searchsorted(time,bs_t0,side='left', sorter=None)        
    bf = np.searchsorted(time,bs_t1,side='left', sorter=None) 
    dataPool_spw = np.zeros((dataPool.shape))


    for i_cond in range(dataPool.shape[0]):
        for i_rep in range(dataPool.shape[1]):
            for i_site in range(dataPool.shape[2]):
                for i_freq in range(dataPool.shape[3]):
                    X = dataPool[i_cond,i_rep,i_site,i_freq,:]
                    baseline = dataPool[:,:,i_site,i_freq,b0:bf]
                    baseline = np.nanmean(baseline,axis=2) # average time
                    X_bs = np.nanmean(baseline.flatten())
                    dataPool_spw[i_cond,i_rep,i_site,i_freq,:] =  ((X-X_bs)/X_bs)*100        
    
    
    print(dataPool_spw.shape)
    if cond == 0:
        n_sess = 10
        sessions = [i for i in range(10)]
    elif cond == 1: 
        n_sess = 11  
        exclude = 7
        sessions = [i for i in range(11)]
        sessions = [x for i,x in enumerate(sessions) if i!=exclude]
        
    chans = np.arange(12).astype(int) 
    freqs =  np.arange(dataPool_spw.shape[3]).astype(int) 
    #print(chans)
    #print(freqs)
    n_jobs = chans.shape[0]*freqs.shape[0]
    print(n_jobs)
    # ACME Job array
    combs = np.zeros((n_jobs,2))
    count = -1
    for ch in range(chans.shape[0]):
        for fr in range(freqs.shape[0]):             
            count += 1
            combs[count] = np.hstack((ch,fr))
    #print(combs.astype(int))
    var = combs[jobs]
    print(var[0])
    print(var[1])
    ch = var[0].astype(int)
    fr = var[1].astype(int)
    print('block: ', conds[cond])
    print('method: ', methods[fband])
    print('frequency band: ', fbands[fband])
    #print('spectral var: ', svars[svar])
    print('sessions :', n_sess)
    print('channel idx :', ch)
    print('frequency idx :', fr)
    # Index channels according to laminar channels
    channels = [i for i in range(dataPool_spw.shape[2])] # Total channels (sessions x sites)
    site_idx = np.zeros((12,n_sess)).astype(np.uint) # Index to to channels (concatenated sessions)
    for n in range(12):
        site = [x for x in channels if x%12 == n]
        site_idx[n,:] = site
    
    if split == 0:
        min_rep = 5
        data_spw = dataPool_spw[:,::2,site_idx[ch,sessions],fr,:]
        #data_gpr = dataPool_gpr[:,::2,site_idx[ch,sessions],fr,:]
        #data = np.append(data_spw,data_gpr,axis=2)
        data = data_spw
    elif split == 1: 
        min_rep = 5 
        data_spw = dataPool_spw[:,1::2,site_idx[ch,sessions],fr,:]
        #data_gpr = dataPool_gpr[:,1::2,site_idx[ch,sessions],fr,:]
        #data = np.append(data_spw,data_gpr,axis=2)
        data = data_spw
    elif split == 2: 
        min_rep = 10 
        data_spw = dataPool_spw[:,:,site_idx[ch,sessions],fr,:]
        #data_gpr = dataPool_gpr[:,:,site_idx[ch,sessions],fr,:]
        #data = np.append(data_spw,data_gpr,axis=2)
        data = data_spw
        
    
    data[np.isnan(data)] = 0
    print(data.shape)
    diss = np.zeros((data.shape[0], data.shape[0], data.shape[3]))
    print(diss.shape)
    # Decoding
    msg = ('Pairwise decoding \n')
    # Performing the analysis independently for each time-point
    np.random.seed()
    for t in tqdm(range(data.shape[3]), desc=msg, position=0): # for timepoints
        for i1 in range(data.shape[0]): # for conditions
            for i2 in range(data.shape[0]): # for conditions
                if i1 <= i2:
                    # Condition 1 selection & shuffling
                    dat_1 = data[i1,:,:,t]
                    dat_1 = dat_1[shuffle(np.arange(len(dat_1))),:]
                    # Condition 2 selection & shuffling
                    dat_2 = data[i2,:,:,t]
                    dat_2 = dat_2[shuffle(np.arange(len(dat_2))),:]
                    score = np.zeros(dat_1.shape[0])
                    # Training and testing the classifier
                    for f in range(len(dat_1)): # for folds
                        # Defining the train/test indices
                        train_idx = np.arange(len(dat_1))
                        train_idx = np.delete(train_idx, f)
                        test_idx = f
                        # Defining the X_train & X_test partitions
                        d_1 = dat_1[train_idx]; d_2 = dat_2[train_idx]
                        X_train = np.append(d_1, d_2, axis=0)
                        d_1 = dat_1[test_idx]; d_1 = np.reshape(d_1, (1,-1))
                        d_2 = dat_2[test_idx]; d_2 = np.reshape(d_2, (1,-1))
                        X_test = np.append(d_1, d_2, axis=0)
                        # Defining the Y_train & Y_test partitions
                        Y_train = np.append(np.zeros((min_rep-1)), np.ones((min_rep-1)))
                        Y_test = np.asarray((0,1))
                        # Shuffling the training data across conditions
                        idx = shuffle(np.arange(len(X_train)))
                        X_train = X_train[idx]
                        Y_train = Y_train[idx]
                        # Defining the classifier
                        #clf = RankSimilarityClassifier()
                        clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
                        #clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0)
                        # Training the classifier
                        scaler = StandardScaler().fit(X_train)
                        X_train = scaler.transform(X_train)
                        X_test = scaler.transform(X_test)
                        clf.fit(X_train, Y_train)
                        # Testing the classifier
                        Y_pred = clf.predict(X_test)
                        score[f] = sum(Y_pred == Y_test)/2
                    # Storing the results
                    acc = sum(score) / dat_1.shape[0]  
                    diss[i1,i2,t] = acc
                    diss[i2,i1,t] = acc
    # Save results
    
    # Split 1  (odd trials)
    if split == 0:
        fname = os.path.join(results_path + svars[0]
                             + '_Dec_ch' + str(ch+1) 
                             + '_freq' + str(fr+1) 
                             + '_' + fbands[fband] 
                             + '_' + methods[fband] 
                             + '_split1_norm_c.npy') 
    # Split 2 (even trials)
    elif split == 1:
        fname = os.path.join(results_path + svars[0]
                             + '_Dec_ch' + str(ch+1) 
                             + '_freq' + str(fr+1) 
                             + '_' + fbands[fband] 
                             + '_' + methods[fband] 
                             + '_split2_norm_c.npy')
    # Whole data (all trials)
    elif split == 2:
        fname = os.path.join(results_path + svars[0]
                             + '_Dec_ch' + str(ch+1) 
                             + '_freq' + str(fr+1) 
                             + '_' + fbands[fband] 
                             + '_' + methods[fband] 
                             + '_norm_c.npy')
    np.save(fname, diss)
    
    return diss
                                                                                                                                              