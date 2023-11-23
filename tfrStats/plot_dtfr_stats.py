import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors

from tfrStats.load_uv_tfrs import load_uv_tfrs as load_uv_tfrs
from tfrStats.get_dpvals_whole import get_dpvals_whole as get_dpvals_whole
from tfrStats.get_dpvals_minmax import get_dpvals_minmax as get_dpvals_minmax
from tfrStats.cluster_correction import cluster_correction as cluster_correction

import warnings
warnings.filterwarnings('ignore')

def plot_dtfr_stats(input_path, cond, fband, null, type):

    """
    Plot empirical TFR and stats results for depth x time TFR

    This functions use load_uv_tfrs, as well as optionally get_pvals_minmax, 
    get_pvals_whole and (also optionally) cluster_correction to plot the empirical TFR,
    the p-values and the corrected threshold. Correction for multiple comparisons is
    already taken into account by get_pvals_minmax and get_pvals_whole. Optionally, 
    cluster_correction corrects the p-values for multiple comparisons using a distance
    threshold for neighbours frequencies and time bins if they are alltogheter above alpha.
    
    
    
     .. todo::  
        * Handle parameters with dictionary. 


    :param string input_path: path to the .npz file.
    :param in condition: condition index (i.e. 0, 1, 2, 3).
    :param int svar: spectral power or GPR (not implemented here).
    :param int fband: frequency band index (i.e. low, high, higher).
    :param int obs: [nullType, percentile], two integeres: 0 for min-max, 1 for whole, 0-100 percentile.
    :param int correction: 1 for p-values, 2 for cluster corrected p-values.
    :param int cluster_size: cluster size.
    :param float alpha: alpha.    
    
    :return: empirical time frequency representation n_conds x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113).
    :return: null time frequency representation (i.e. 30, 12, 16, 113  or 1000, 30, 12, 16, 2).
    :rtype: float
 
    @author: Nicolas Gravel, 19.09.2023 
    """
        
    tps           = [57,113,141,140] # time windows
    fps           = [19,16,11,1] # frequency bins
    lp            = [2, 20, 80] # low cut
    hp            = [20, 80, 200] # high cut
    twindow       = [65, 80] # window in the plot
    stats_range   = [400, 1000] # range for thresholding (interval or "cluster" to compute the threshold)
    #stats_range   = [0, 400] # range for thresholding (interval or "cluster" to compute the threshold)
    ups           = 4 # upsampling in figure
    cmap          = 'cubehelix_r'
    cnorm         = 0
    coloff        = 0.5        # colormap center offset
    cnorm_range   = [400, 1000]
    maxpwr        = 150
    overlay_range = [1,-1]  # range for overlay coverage
    calpha         = 0.25
    prctl         = null[0]
    alpha         = null[1]


    if type == 'minmax' or type == 'minmax_oll':
        results       = 1
        tfr_emp, tfr_null = load_uv_tfrs(input_path, [], cond, fband, results) # load tfrs from .npz file

    if type == 'whole' or type == 'whole_roll':
        results       = 0
        tfr_emp, tfr_null = load_uv_tfrs(input_path, [], cond, fband, results) # load tfrs from .npz file
    



    ## helper function used by plot_stats to noramlize colormap ranges
    def coloroffset(min_val, max_val, k):

        if 0 <= k <= 1:  # Ensure k is between 0 and 1
            point = min_val + k*(max_val - min_val)
            #print(f'For k={k}, the point in the range {min_val}-{max_val} is: {point}')
        #else:
            #print("Error: k must be between 0 and 1") 
        return point
    

    ## Plot TFR across sites
    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(6,4))



    # indices for plotting
    x =  np.linspace(start = -800, stop = 2000, num = tps[fband])# time vector
    t0 = np.searchsorted(x, stats_range[0],side='left', sorter=None) # time index for induced power period start
    tf = np.searchsorted(x, stats_range[1],side='left', sorter=None) # time index for induced power period end
    y = np.linspace(lp[fband], hp[fband], fps[fband])
    y2 = np.linspace(lp[fband], hp[fband], fps[fband]*ups)
    X, Y = np.meshgrid(x, y)
    x2 =  np.linspace(start = -800, stop = 2000, num = 280)
    X2, Y2 = np.meshgrid(x2, y2)

    # plot empirical TFR
    tfr_emp_ = np.squeeze(np.nanmean(tfr_emp,axis=0))
    gavg = np.squeeze(np.nanmean(tfr_emp_,axis=0))
    gavg[np.isnan(gavg)] = 0



    # Cross-frequency TFR
    x    = np.linspace(start = -800, stop = 2000, num = tps[fband])
    tt0  = np.searchsorted(x,stats_range[0],side='left', sorter=None)
    ttf  = np.searchsorted(x,stats_range[1],side='left', sorter=None)
    pwr  = np.mean(gavg[:,tt0:ttf],axis=1)
    #print(pwr.shape)
    x    = np.linspace(lp[fband], hp[fband], num = fps[fband])
    if fband == 0:
        peak  = np.argmax(pwr); 
        #print(peak)
        sigma = 2
    if fband == 2:
        peak  = np.argmax(pwr[0:5]); 
        #print(peak)
        sigma = 2
    else:
        peak   = np.argmax(pwr); 
        sigma  = 2
    pk    = peak.astype(int)
    #print('peak frequency : ', x[pk])
    #print('peak power :', pwr[pk])
    #peaks[contrast_idx,fband,0] = x[pk]

    #tfr emp  :  (30, 12, 16, 113)   
    #tfr null    (1000, 30, 12, 16, 2)

    if pk-sigma<=0: 
        pwr_avg = np.mean(pwr[pk:pk+2*sigma])
        print('peak frequency range : ', x[pk+2*sigma])
        print('power average within peak:', pwr_avg)
        davg = np.squeeze(np.nanmean(tfr_emp[:,:,pk:pk+2*sigma,:],axis=0))
        if type == 'minmax':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,:,pk:pk+2*sigma,:],axis=1))
        if type == 'whole':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,pk:pk+2*sigma,:],axis=0))

    elif pk+sigma>=fps[fband]: 
        pwr_avg = np.mean(pwr[pk-2*sigma:pk])
        print('peak frequency range : ', x[pk-2*sigma])
        print('power average within peak:', pwr_avg)
        davg = np.squeeze(np.nanmean(tfr_emp[:,:,pk-2*sigma:pk,:],axis=0))
        if type == 'minmax':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,:,pk-2*sigma:pk,:],axis=1))
        if type == 'whole':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,pk-2*sigma:pk,:],axis=0))
  
    else:
        print('peak frequency range : ', x[pk-sigma], x[pk+sigma])
        pwr_avg = np.mean(pwr[pk-sigma:pk+sigma])
        print('power average within peak:', pwr_avg)
        #print(tfr_emp.shape)
        davg = np.squeeze(np.nanmean(tfr_emp[:,:,pk-sigma:pk+sigma,:],axis=0))
        if type == 'minmax':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,:,pk-sigma:pk+sigma,:],axis=1))
        if type == 'whole':
            davg_null = np.squeeze(np.nanmean(tfr_null[:,:,pk-sigma:pk+sigma,:],axis=0))

    #print(davg.shape, davg_null.shape)

    if type == 'minmax':
        print('min-max')
        stats = get_dpvals_minmax(davg, davg_null, tail = 'single-sided')
    if type == 'whole':
        print('whole-null')
        stats = get_dpvals_whole(davg, davg_null, fband)


    davg[np.isnan(davg)] = 0
    davg_null[np.isnan(davg_null)] = 0
    #print('depth average  :', davg.shape)
    #print('depth null-average  :', davg_null.shape) # e.g. depth null-average  : (1000, 12, 4, 2)

    x =  np.linspace(start = -800, stop = 2000, num = tps[fband])
    y = np.linspace(start=-550, stop=550, num=12).astype(int)
    x2 =  np.linspace(start = -800, stop = 2000, num = 280)
    y2 = np.linspace(start=-550, stop=550, num=12*ups).astype(int)
    X, Y = np.meshgrid(x, y)
    X2, Y2 = np.meshgrid(x2, y2)

    #print(davg.shape)
    davg = np.mean(davg,axis=1) 
    f = interp2d(x, y, np.flipud(davg), kind='linear')

    # Color map normalization
    tt0 = np.searchsorted(x,cnorm_range[0],side='left', sorter=None)
    ttf = np.searchsorted(x,cnorm_range[1],side='left', sorter=None)
    tfrange = davg[:,tt0:ttf] 
    _min = np.min(np.min(tfrange.flatten()))
    _max = np.max(np.max(tfrange.flatten()))
    #print('min =',_min,'max =',_max)   
    if cnorm == 1:
        vmin = _min  
        vmax =  maxpwr
        if fband== 2:
            vmax = maxpwr/3
    elif cnorm == 0 :
        vmin = _min  
        vmax = _max
    vcenter = coloroffset(vmin, vmax, coloff)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)  
    TFR_emp = f(x2, y2)
    im_spwr = ax[0].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_emp[:,twindow[0]:-twindow[1]] , cmap=cmap,norm=norm)
    
    #f_null = interp2d(x, y, np.flipud(davg_null), kind='linear')

    # Thresholding using truncated min-max distribution
    if type == 'minmax':
        h0       = davg_null[:,:,:,1] 
        h0       = np.amax(h0, axis=2) # max across frequency bins
        h0       = np.amax(h0, axis=1) # max across depths
        davg_thr = np.percentile(h0.flatten(),prctl) 
        print('cutoff computed using min/max of null distribution: ', davg_thr )


    # Thresholding using whole distribution
    if type == 'whole':
        t0 = np.searchsorted(x,stats_range[0],side='left', sorter=None)
        td = np.searchsorted(x,stats_range[1],side='left', sorter=None) 
        null = np.mean(davg_null,axis=0)
        null[1:-1,0:t0]  = np.nan
        null[1:-1,td:-1] = np.nan
        null_ = null[~np.isnan(null)]
        davg_thr = np.percentile(null_.flatten(),prctl)
        print('cutoff computed using whole null distribution: ', davg_thr )
    
    cut = np.full((TFR_emp.shape[0],TFR_emp.shape[1]),davg_thr)
    t0 = np.searchsorted(x2,stats_range[0],side='left', sorter=None)
    td = np.searchsorted(x2,stats_range[1],side='left', sorter=None)
    cut[1:-1,0:t0] = np.nan
    cut[1:-1,td:-1] = np.nan
    
    THR =  TFR_emp >= cut
    significant = THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]]*TFR_emp[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]]
    im_pvals = ax[0].pcolormesh(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        significant, cmap=cmap,norm=norm,alpha=calpha)

    ax[1].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        origin='upper',
                        colors='red',
                        linestyles='solid',
                        linewidths=0.5)


    # Plot p-values ad thresholding

    cut = np.full((TFR_emp.shape[0],TFR_emp.shape[1]),alpha)
    t0 = np.searchsorted(x2,stats_range[0],side='left', sorter=None)
    td = np.searchsorted(x2,stats_range[1],side='left', sorter=None)
    cut[1:-1,0:t0] = np.nan
    cut[1:-1,td:-1] = np.nan


    f = interp2d(x, y, np.flipud(stats), kind='linear')
    TFR_pvals = f(x2, y2)
    THR = TFR_pvals <= cut #alpha
    im_pvals = ax[1].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_pvals[:,twindow[0]:-twindow[1]])
    ax[0].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                    THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                    origin='upper',
                    colors='dodgerblue',
                    linestyles='solid',
                    linewidths=0.5)


    cbar = plt.colorbar(im_spwr,cax = fig.add_axes([0.95, 0.6, 0.02, 0.15]),extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('power (%)',fontsize=10)
    cbar = plt.colorbar(im_pvals,cax = fig.add_axes([0.95, 0.2, 0.02, 0.15]),extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('p-value',fontsize=10)
    ax[1].set_xlabel('Time (ms)', fontsize=12)
    ax[0].set_ylabel('depth (um)', rotation=90, fontsize=10)
    ax[1].set_ylabel('depth (um)', rotation=90, fontsize=10)
    ax[0].title.set_text('power % change relative to baseline')
    #ax[1].title.set_text('p-values')
    txt='Cutoff (blue outline) is valid for the 400-1000 ms window.'
    fig.text(0.5, -0.06, txt, ha='center')

    # Figure params
    blocks  = ['grat', 'nat']
    svars   = ['spw', 'gpr']
    fbands  = ['low','high','higher']  
    svar = 0
    if cond == 0:
        fname = str(input_path +'uvtfr_dtfr_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + '.png')
    if cond == 1:
        fname = str(input_path +'uvtfr_dtfr_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + '.png')
    if cond == 2:
        fname = str(input_path +'uvtfr_dtfr_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + '.png')
    if cond == 3:
        fname = str(input_path +'uvtfr_dtfr_' +    fbands[fband]  + '_' + blocks[cond] + '_' + svars[svar] + '.png')

    print('figure :', fname)
    plt.savefig(fname, bbox_inches="tight")

    return TFR_emp, significant, THR


