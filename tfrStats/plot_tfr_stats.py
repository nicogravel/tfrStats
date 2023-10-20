import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors

# function to plot the tfr + stats rresults loaded using load_uv_tfr s
def plot_tfr_stats(input_path, cond, fband, null, correction, cluster_size, type, alpha):

    """
    Plot empirical TFR and stats results

    This functions use load_uv_tfrs, as well as optionally get_pvals_minmax, 
    get_pvals_whole and (also optionally) cluster_correction to plot the empirical TFR,
    the p-values and the corrected threshold. Correction for multiple comparisons is
    already taken into account by get_pvals_minmax and get_pvals_whole. Optionally, 
    cluster_correction corrects the p-values for multiple comparisons using a distance
    threshold for neighbours frequencies and time bins if they are alltogheter above alpha.
    
    
    Args:
        input path: string 
        condition: an integer
        null: [nullType, percentile], two integeres: 0 for min-max, 1 for whole, 0-100 percentile
        fband: frequency band index (i.e. low, high, higher), integer
        cluster size: integer
        correction: integer, 1 p-values, 2 for cluster corrected p-values
        alpha: alpha... float

    Returns:
        
        In case of min-max:
        empirical tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )
        null tfr: n_perm x n_conds x n_sites x frequencies x min/max (i.e. 1000, 30, 12, 16, 2 )

        In case of whole-null
        empirical tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )
        null tfr: n_condss x n_sites x n_freqs x n_time (i.e. 30, 12, 16, 113 )

    @author: Nicolás Gravel, 19.09.2023  
    
    https://nicogravel.github.io/
    
    """
        
    tps           = [57,113,141,140] # time windows
    fps           = [19,16,11,1] # frequency bins
    lp            = [2, 20, 80] # low cut
    hp            = [20, 80, 200] # high cut
    twindow       = [65, 80] # window in the plot
    stats_range   = [400, 1000] # range for thresholding (interval or "cluster" to compute the threshold)
    ups           = 4 # upsampling in figure
    cmap          = 'cubehelix_r'
    cnorm         = 0
    coloff        = 0.5        # colormap center offset
    cnorm_range   = [400, 1000]
    maxpwr        = 150
    overlay_range = [1,-1]  # range for overlay coverage
    calpha         = 0.25
    prctl    = null[1]
    results  = null[0]

    ## Plot TFR across sites
    fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(6,4))

    if type == 'minmax':
        ## Min-max

        tfr_emp, tfr_null = load_uv_tfrs(input_path, [], cond, fband, results) # load tfrs from .npz file

        if correction == 1:
            stats = get_pvals_minmax(tfr_emp, tfr_null, tail = 'single-sided')

        if correction == 2:
            stats = get_pvals_minmax(tfr_emp, tfr_null, tail = 'single-sided')
            pvals_corr = cluster_correction(stats, tfr_null, cluster_size, alpha)

    if type == 'whole':

        tfr_emp, tfr_null = load_uv_tfrs(input_path, [], cond, fband, results)

        if correction == 1:
            stats = get_pvals_whole(tfr_emp, tfr_null, fband)

        if correction == 2:
            stats = get_pvals_whole(tfr_emp, tfr_null, fband)
            pvals_corr = cluster_correction(stats, tfr_null, cluster_size, alpha)

    # indices for plotting
    x =  np.linspace(start = -800, stop = 2000, num = tps[fband])# time vector
    t0 = np.searchsorted(x, 400,side='left', sorter=None) # time index for induced power period start
    tf = np.searchsorted(x, 1000,side='left', sorter=None) # time index for induced power period end
    y = np.linspace(lp[fband], hp[fband], fps[fband])
    y2 = np.linspace(lp[fband], hp[fband], fps[fband]*ups)
    X, Y = np.meshgrid(x, y)
    x2 =  np.linspace(start = -800, stop = 2000, num = 280)
    X2, Y2 = np.meshgrid(x2, y2)

    # plot empirical TFR
    tfr_emp = np.squeeze(np.nanmean(tfr_emp,axis=0))
    gavg = np.squeeze(np.nanmean(tfr_emp,axis=0))
    gavg[np.isnan(gavg)] = 0
    f = interp2d(x, y,gavg, kind='linear')

    # normalize colormap
    tt0 = np.searchsorted(x,cnorm_range[0],side='left', sorter=None)
    ttf = np.searchsorted(x,cnorm_range[1],side='left', sorter=None)
    tfrange = gavg[:,tt0:ttf]
    _min = np.min(np.min(tfrange.flatten()))
    _max = np.max(np.max(tfrange.flatten()))
    print('min =',_min,'max =',_max)
    if cnorm == 1:
        vmin = _min
        vmax =  maxpwr
        if fband == 2:
                vmax = maxpwr/3
    elif cnorm == 0 :
        vmin = _min
        vmax = _max
    vcenter = coloroffset(vmin, vmax, coloff)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    TFR_emp = f(x2, y2)
    im_spwr = ax[0].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_emp[:,twindow[0]:-twindow[1]],cmap=cmap, norm=norm)




    # Thresholding using truncated min-max distribution
    if type == 'minmax':
        h0       = np.nanmean(np.nanmean(tfr_null[:,:,:,:,1],axis=1),axis=1) # average conditions and sites
        print('H0 dimensons :', h0.shape)
        gavg_thr = np.percentile(h0.flatten(),prctl) # pool permutations for all frequencies
        print('cutoff computed using truncated min/max of null distribution: ', gavg_thr )

    # Thresholding using whole distribution
    if type == 'whole':
        gavg_null = np.squeeze(np.nanmean(tfr_null,axis=0))
        gavg_null[np.isnan(gavg_null)] = 0. # just for plotting
        t0 = np.searchsorted(x2,stats_range[0],side='left', sorter=None)
        td = np.searchsorted(x2,stats_range[1],side='left', sorter=None)
        null = gavg_null
        null[1:-1,0:t0]  = np.nan
        null[1:-1,td:-1] = np.nan
        null_ = null[~np.isnan(null)]
        gavg_thr = np.percentile(null_.flatten(),prctl)
        print('cutoff computed using whole null distribution: ', gavg_thr )
    cut = np.full((TFR_emp.shape[0],TFR_emp.shape[1]),gavg_thr)
    t0 = np.searchsorted(x2,stats_range[0],side='left', sorter=None)
    td = np.searchsorted(x2,stats_range[1],side='left', sorter=None)
    cut[1:-1,0:t0] = np.nan
    cut[1:-1,td:-1] = np.nan
    THR =  TFR_emp >= cut
    significant = THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]]*TFR_emp[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]]
    im_pvals = ax[0].pcolormesh(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        significant, cmap=cmap,norm=norm,alpha=calpha)

    ax[0].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        origin='upper',
                        colors='dodgerblue',
                        linestyles='solid',
                        linewidths=0.5)


    # Plot p-values ad thresholding
    if correction == 1:
        f = interp2d(x, y, stats, kind='linear')
        TFR_pvals = f(x2, y2)
        THR = TFR_pvals <= alpha
        im_pvals = ax[1].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_pvals[:,twindow[0]:-twindow[1]])
        ax[1].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                        origin='upper',
                        colors='dodgerblue',
                        linestyles='solid',
                        linewidths=0.5)
    # Cluster
    elif correction == 2:
        f = interp2d(x, y, stats, kind='linear')
        TFR_pvals = f(x2, y2)
        f = interp2d(x, y, pvals_corr, kind='linear')
        TFR_pvals_corr = f(x2, y2)
        THR = TFR_pvals_corr <= alpha
        im_pvals = ax[1].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_pvals[:,twindow[0]:-twindow[1]])
        ax[1].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
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
    ax[0].set_ylabel('frequency (Hz)', rotation=90, fontsize=10)
    ax[1].set_ylabel('frequency (Hz)', rotation=90, fontsize=10)
    ax[0].title.set_text('power % change relative to baseline')
    #ax[1].title.set_text('p-values')
    txt='Cutoff (blue outline) is valid for the 400-1000 ms window.'
    fig.text(0.5, -0.06, txt, ha='center')
    return