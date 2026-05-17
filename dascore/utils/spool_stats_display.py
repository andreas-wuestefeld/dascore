# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import dascore as dc

import pandas as pd
from scipy import stats



tmpfile = Path(r"O:\Staff\andreasw\Dev\FibreEyes\Aurland\_dascore_index_Aurland.hdf5")
tmpfile = Path(r"O:\Staff\andreasw\Dev\FibreEyes\Aurland\_dascore_index_Aurland_dataframe.plk")


df = pd.read_pickle(tmpfile)
subset = df[df['time_min'] > np.datetime64('2026-01-16T14:00:00')]

#%%
def ransac(points, iterations=500, threshold=None, min_inliers=20, sample_size=10, mad_scale=3.0):
    n = len(points)
    if sample_size > n:
        raise ValueError("sample_size cannot be larger than number of points")

    best_mask = None
    best_count = 0
    switch_iter = max(1, int(0.1 * iterations))

    for it in range(iterations):
        idx = np.random.choice(n, sample_size, replace=False)
        sample = points[idx]

        x = sample[:, 0]
        y = sample[:, 1]

        # fit y = m*x + b
        m, b = np.polyfit(x, y, 1)

        # vertical residuals are fine for nearly horizontal lines
        d = np.abs(points[:, 1] - (m * points[:, 0] + b))

        # loose threshold before robust estimate is available
        t = threshold if threshold is not None else np.median(d) * 2.0

        mask = d < t
        count = np.sum(mask)

        if count > best_count and count >= min_inliers:
            best_mask = mask
            best_count = count

        # estimate threshold after 10% of iterations from current best model
        if it == switch_iter and threshold is None and best_mask is not None:
            inliers = points[best_mask]
            m0, b0 = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
            d0 = np.abs(points[:, 1] - (m0 * points[:, 0] + b0))

            # use current inliers only for robust scale
            d_in = d0[best_mask]
            med = np.median(d_in)
            mad = np.median(np.abs(d_in - med))
            sigma = 1.4826 * mad
            threshold = mad_scale * sigma

            if threshold == 0:
                threshold = max(np.std(d_in), 1e-6)

    if best_mask is None:
        return None, None, None, None

    # final refinement on all best inliers
    inliers = points[best_mask]
    m, b = np.polyfit(inliers[:, 0], inliers[:, 1], 1)

    d = np.abs(points[:, 1] - (m * points[:, 0] + b))
    mask = d < threshold

    #inliers = points[mask]
    #outliers = points[~mask]

    # refine once more after final classification
    #if len(inliers) >= 2:
    #    m, b = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
    outlier_idx = np.where(mask==0)
    return outlier_idx #(m, b), inliers, outliers, threshold





#%%
def viz_spool(df, method='mode',  tolerance_percent=20, iterations=500):
    filetimes = df['time_min'].to_numpy()
    gap       = np.diff(filetimes) / np.timedelta64(1,'s')
    fsamp     = 1 / (df['time_step'].to_numpy() / np.timedelta64(1, 's'))


    def _find_outliers(gap, method,  tolerance_percent, iteration):
        def _is_not_within_tolerance(value, reference, tolerance_percent):
            # Calculate the absolute difference as a fraction of the reference
            diff_percent = abs((value - reference) / reference)

            # Returns True if the difference is tolerance_percent  or less
            return diff_percent > tolerance_percent/100


        if method == 'mode':
            #most common value; outliers are exact not matches!
            value = stats.mode(gap).mode
            outlier_index = np.where(gap != value)[0]

        elif method == 'mean':
            reference_value = np.mean(gap)
            outlier_index = np.where(_is_not_within_tolerance(gap,reference_value, tolerance_percent ))[0]

        elif method == 'median':
            reference_value = np.median(gap)
            outlier_index = np.where(_is_not_within_tolerance(gap,reference_value, tolerance_percent ))[0]

        elif method == 'ransac': #this is slow , but accurate
            sample_size = int(gap.shape[0] / 20) #20% of data should be good
            data = np.vstack((filetimes[1:].astype(np.int32), gap)).T
            outlier_index = ransac(
                data,
                iterations=iterations,
                sample_size=sample_size,
                min_inliers=30
            )
        else:
            raise Exception(f'Unknown outlider detection method {method}. Allowed are ["mode", "mean", "median", "ransac"]')

        return outlier_index


    outlier_index = _find_outliers(gap, method,  tolerance_percent, iterations)



    #plotting
    tick_map = [
        [       1, '1 sec'],
        [      10, '10 sec'],
        [      60, '1 min'],
        [   10*60, '10 min'],
        [    3600, '1 hour'],
        [  6*3600, '6 hours'],
        [   86400, '1 day'],
        [ 7*86400, '1 week'],
        [30*86400, '1 month']
             ]
    ticks      = list(zip(*tick_map))[0]
    ticklabels = list(zip(*tick_map))[1]


    #% Plot
    fig, axs = plt.subplots(2,1, figsize=(12,10), layout='constrained')

    # Plot file-time differences
    ax = axs[0]
    ax.semilogy(filetimes[:-1], gap, '.')
    ax.set_yticks(ticks, ticklabels, )
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid('on')
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel('Filetime Difference')

    for i in outlier_index:
        txt = str(np.timedelta64(int(gap[i]),'s').item())
        txt = '  ' + str(filetimes[i])[:19]
        ax.text(filetimes[i], gap[i], txt, rotation=90, ha='center', fontsize=8);

    # add start label
    txt = str(filetimes[0])[:19].replace('T','\n')
    ax.text(filetimes[0], gap[0],   txt, rotation=90,  va='center', ha='right',  fontsize=8, color='darkgreen');
    #add end label
    txt = str(filetimes[-1])[:19].replace('T','\n')
    ax.text(filetimes[-1], gap[-1], txt, rotation=90, va='center', ha='left',  fontsize=8, color='red');
    ax.set_title('Filetime Difference')



    #plor sampling rate evolution
    ax = axs[1]
    ax.plot(filetimes, fsamp, '-', lw=3)
    #ax.set_yticks(ticks, ticklabels, )
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid('on')
    #ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel('Sample Rate [Hz]')
    ax.set_title('Sample Rate Evolution')

    return ax, outlier_index

ax,idx = viz_spool(subset)
