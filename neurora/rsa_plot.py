# -*- coding: utf-8 -*-

' a module for plotting the RDM '

__author__ = 'Zitong Lu'

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def plot_rdm_1(rdm):

    if len(np.shape(rdm)) != 2:

        return None

    a, b = np.shape(rdm)

    if a != b:

        return None

    plt.imshow(rdm, cmap=plt.cm.jet, clim=(0, 1))

    plt.axis("off")

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Dissimilarity", fontdict=font)

    plt.show()

def plot_rdm_2(rdm):

    if len(np.shape(rdm)) != 2:

        return None

    a, b = np.shape(rdm)

    if a != b:

        return None

    plt.imshow(rdm, cmap=plt.cm.Greens, clim=(0, 1))

    plt.axis("off")

    for i in range(a):
        for j in range(b):
            text = plt.text(i, j, float('%.4f'%rdm[i, j]),
                           ha="center", va="center", color="blue")

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Dissimilarity", fontdict=font)

    plt.show()

def plot_corrs_by_time(corrs, labels, time_unit=[0, 0.1]):
    # corrs represent the correlation coefficients point-by-point, its shape :
    #       [n, ts, 2] (here 2 contains r-value and p-value) or [n, ts] (r-value)
    # label represent the conditions of RSA results, its shape : [n]
    # time_unit=[start_t, t_step]

    n = corrs.shape[0]
    ts = corrs.shape[1]

    start_t = time_unit[0]
    tstep = time_unit[1]

    end_t = start_t + ts * tstep

    x = np.arange(start_t, end_t, tstep)

    t = ts * 50

    x_soft = np.linspace(x.min(), x.max(), t)

    y_soft = np.zeros([n, t])

    for i in range(n):
        if len(corrs.shape) == 3:
            y_soft[i] = spline(x, corrs[i, :, 0], x_soft)
        if len(corrs.shape) == 2:
            y_soft[i] = spline(x, corrs[i, :], x_soft)

    fig, ax = plt.subplots()

    for i in range(n):
        if labels:
            plt.plot(x_soft, y_soft[i], linewidth=4, label=labels[i])
        else:
            plt.plot(x_soft, y_soft[i], linewidth=4)

    plt.ylim(0, 1)
    plt.ylabel("Similarity", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    plt.tick_params(labelsize=18)

    if labels:
        plt.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

def plot_corrs_hotmap(eegcorrs, chllabels=[], time_unit=[0, 0.1], smooth=True):
    # eegcorrs represents the correlation coefficients time-by-time, its shape:
    # [N_chls, ts, 2] or [N_chls, ts], N_chls: number of channels, ts: number of time points, 2: a r-value and a p-value
    # chllabel represents the names of channels
    # time_unit=[start_t, t_step]
    # smooth represents smoothing the results or not

    nchls = eegcorrs.shape[0]
    ts = eegcorrs.shape[1]

    start_t = time_unit[0]
    tstep = time_unit[1]

    end_t = start_t + ts * tstep

    x = np.arange(start_t, end_t, tstep)

    for i in range(nchls):
        if i % 10 == 0 and i != 10:
            newlabel = str(i+1) + "st"
        elif i % 10 == 1 and i != 11:
            newlabel = str(i+1) + "nd"
        elif i % 10 == 2 and i != 12:
            newlabel = str(i+1) + "rd"
        else:
            newlabel = str(i+1) + "th"
        chllabels.append(newlabel)

    if smooth == True:

        t = ts * 50

        x_soft = np.linspace(x.min(), x.max(), t)

        y_soft = np.zeros([nchls, t])

        for i in range(nchls):
            if len(eegcorrs.shape) == 3:
                y_soft[i] = spline(x, eegcorrs[i, :, 0], x_soft)
            elif len(eegcorrs.shape) == 2:
                y_soft[i] = spline(x, eegcorrs[i, :], x_soft)

        rlts = y_soft

    if smooth == False:
        if len(eegcorrs.shape) == 3:
            rlts = eegcorrs[:, :, 0]
        elif len(eegcorrs.shape) == 2:
            rlts = eegcorrs

    print(rlts.shape)

    fig = plt.gcf()
    fig.set_size_inches(10, 3)

    plt.imshow(rlts, extent=(start_t*nchls/3, end_t*nchls/3, 0, 0.16*nchls), clim=(0, 1), origin='low')

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Similarity", fontdict=font)
    xi = []
    for i in range(nchls):
        xi.append(0.16*i + 0.08)
    yi = chllabels
    plt.tick_params(labelsize=18)
    plt.yticks(xi, yi, fontsize=18)
    plt.ylabel("Channel", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    plt.show()