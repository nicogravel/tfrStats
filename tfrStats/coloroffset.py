import numpy as np
from tqdm.auto import tqdm
import scipy.io as sio
from numpy import inf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.colors as colors



## helper function used by plot_stats to noramlize colormap ranges
def coloroffset(min_val, max_val, k):
    """
    Set colormap center offset

    Helper function used by plot_stats to noramlize colormap ranges
    
    """
    if 0 <= k <= 1:  # Ensure k is between 0 and 1
        point = min_val + k*(max_val - min_val)
        #print(f'For k={k}, the point in the range {min_val}-{max_val} is: {point}')
    #else:
        #print("Error: k must be between 0 and 1")

    return point
