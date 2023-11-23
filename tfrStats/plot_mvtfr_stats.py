import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors
from tfrStats.stats_tfrs_avg import stats_tfrs_avg as stats_tfrs_avg  
from tfrStats.cluster_correction import cluster_correction as cluster_correction 


import warnings
warnings.filterwarnings('ignore')

def plot_mvtfr_stats(cond, tfr, tfr_null, fband, alpha, correction):

    """
    Plot empirical Multi-variate TFR and stats results

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
    ups           = 4 # upsampling in figure
    cmap          = 'cubehelix_r'
    cnorm         = 1
    coloff        = 0.5
    overlay_range = [1,-1]  # range for overlay coverage
    alpha         = 0.05


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
    y = np.linspace(lp[fband], hp[fband], fps[fband])
    y2 = np.linspace(lp[fband], hp[fband], fps[fband]*ups)
    X, Y = np.meshgrid(x, y)
    x2 =  np.linspace(start = -800, stop = 2000, num = 280)
    X2, Y2 = np.meshgrid(x2, y2)

    # plot empirical TFR
    tfr_emp = np.squeeze(tfr[:,:,:])
    gavg = np.squeeze(np.nanmean(tfr_emp,axis=0))
    gavg[np.isnan(gavg)] = 0
    print(gavg.shape)
    f = interp2d(x, y,gavg, kind='linear')
    if cnorm == 1 :
        vmin = 0   #_min
        vmax = 0.6 #_max
    vcenter = coloroffset(vmin, vmax, coloff)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    TFR_emp = f(x2, y2)
    im_spwr = ax[0].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_emp[:,twindow[0]:-twindow[1]],cmap=cmap, norm=norm)

    stats = stats_tfrs_avg(tfr_null,tfr, correction)
    f = interp2d(x, y, stats, kind='linear')
    TFR_pvals = f(x2, y2)

    THR = TFR_pvals  <= alpha  #alpha
    im_pvals = ax[1].pcolormesh(X2[:,twindow[0]:-twindow[1]], Y2[:,twindow[0]:-twindow[1]], TFR_pvals[:,twindow[0]:-twindow[1]])

    ax[0].contour(X2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]], Y2[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                    THR[overlay_range[0]:overlay_range[1],twindow[0]:-twindow[1]],
                    origin='upper',
                    colors='dodgerblue',
                    linestyles='solid',
                    linewidths=0.5)

    cbar = plt.colorbar(im_spwr,cax = fig.add_axes([0.95, 0.6, 0.02, 0.15]),extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Spearman rho',fontsize=10)
    cbar = plt.colorbar(im_pvals,cax = fig.add_axes([0.95, 0.2, 0.02, 0.15]),extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('p-value',fontsize=10)
    ax[1].set_xlabel('Time (ms)', fontsize=12)
    ax[0].set_ylabel('frequency (Hz)', rotation=90, fontsize=10)
    ax[1].set_ylabel('frequency (Hz)', rotation=90, fontsize=10)
    ax[0].title.set_text('RDM reliability obtained using different stimulus choices')
    #ax[1].title.set_text('p-values')
    txt=str('Cutoff (blue outline) is corrected across ' + correction)
    fig.text(0.5, -0.06, txt, ha='center')



    return

