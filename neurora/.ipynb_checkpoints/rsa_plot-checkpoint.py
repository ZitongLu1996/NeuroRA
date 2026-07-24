# -*- coding: utf-8 -*-

' a module for plotting the NeuroRA results '

__author__ = 'Zitong Lu'

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import ttest_1samp, ttest_rel
from nilearn import plotting, datasets, surface
import nibabel as nib
from neurora.stuff import get_affine, get_bg_ch2, get_bg_ch2bet, correct_by_threshold, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided, \
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, smooth_1d
from decimal import Decimal


' a function for plotting the RDM '

def plot_rdm(rdm, percentile=False, rescale=False, lim=[0, 1], conditions=None, con_fontsize=12, cmap=None, title=None,
             title_fontsize=16):

    """
    Plot the RDM

    Parameters
    ----------
    rdm : array or list [n_cons, n_cons]
        A representational dissimilarity matrix.
    percentile : bool True or False. Default is False.
        Rescale the values in RDM or not by displaying the percentile.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    conditions : string-array or string-list. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'jet'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(rdm)) != 2 or np.shape(rdm)[0] != np.shape(rdm)[1]:

        return "Invalid input!"

    # get the number of conditions
    cons = rdm.shape[0]

    crdm = copy.deepcopy(rdm)

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    # determine if it's a square
    a, b = np.shape(crdm)
    if a != b:
        return None

    if percentile == True:

        v = np.zeros([cons * cons, 2])
        for i in range(cons):
            for j in range(cons):
                v[i * cons + j, 0] = crdm[i, j]

        index = np.argsort(v[:, 0])
        m = 0
        for i in range(cons * cons):
            if i > 0:
                if v[index[i], 0] > v[index[i - 1], 0]:
                    m = m + 1
                v[index[i], 1] = m

        v[:, 0] = v[:, 1] * 100 / m

        for i in range(cons):
            for j in range(cons):
                crdm[i, j] = v[i * cons + j, 0]

        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 100))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(0, 100))

    # rescale the RDM
    elif rescale == True:

        # flatten the RDM
        vrdm = np.reshape(rdm, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        crdm[i, j] = float((crdm[i, j] - minvalue) / (maxvalue - minvalue))

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    else:

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}

    if percentile == True:
        cb.set_label("Dissimilarity (percentile)", fontdict=font)
    elif rescale == True:
        cb.set_label("Dissimilarity (Rescaling)", fontdict=font)
    else:
        cb.set_label("Dissimilarity", fontdict=font)

    if conditions != None:
        print("1")
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.title(title, fontsize=title_fontsize)

    plt.show()

    return 0


' a function for plotting the RDM with values '

def plot_rdm_withvalue(rdm, lim=[0, 1], value_fontsize=10, conditions=None, con_fontsize=12, cmap=None, title=None,
                       title_fontsize=16):

    """
    Plot the RDM with values

    Parameters
    ----------
    rdm : array or list [n_cons, n_cons]
        A representational dissimilarity matrix.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    value_fontsize : int or float. Default is 10.
        The fontsize of the values on the RDM.
    conditions : string-array or string-list or None. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'Greens'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(rdm)) != 2 or np.shape(rdm)[0] != np.shape(rdm)[1]:

        return "Invalid input!"

    # get the number of conditions
    cons = rdm.shape[0]

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    crdm = copy.deepcopy(rdm)

    # determine if it's a square
    a, b = np.shape(crdm)
    if a != b:
        return None

    # plot the RDM
    min = lim[0]
    max = lim[1]
    if cmap == None:
        plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=plt.cm.Greens, clim=(min, max))
    else:
        plt.imshow(crdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Dissimilarity", fontdict=font)

    # add values
    step = float(1 / cons)
    for i in range(cons):
        for j in range(cons):
            plt.text(i * step + 0.5 * step, 1 - j * step - 0.5 * step, float('%.4f' % rdm[i, j]),
                            ha="center", va="center", color="blue", fontsize=value_fontsize)

    if conditions != None:
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.title(title, fontsize=title_fontsize)

    plt.show()

    return 0


' a function for plotting the correlation coefficients by time sequence '

def plot_corrs_by_time(corrs, labels=None, time_unit=[0, 0.1], title=None, title_fontsize=16, labelpad=0):

    """
    plot the correlation coefficients by time sequence

    corrs : array
        The correlation coefficients time-by-time.
        The shape of corrs must be [n, ts, 2] or [n, ts]. n represents the number of curves of the correlation
        coefficient by time sequence. ts represents the time-points. If shape of corrs is [n, ts 2], each time-point
        of each correlation coefficient curve contains a r-value and a p-value. If shape is [n, ts], only r-values.
    label : string-array or string-list or None. Default is None.
        The label for each corrs curve.
        If label=None, no legend in the figure.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    labelpad : int or float. Default is 0.
        Distance of ylabel from the y-axis.
    """

    if len(np.shape(corrs)) < 2 or len(np.shape(corrs)) > 3:

        return "Invalid input!"

    # get the number of curves
    n = corrs.shape[0]

    # get the number of time-points
    ts = corrs.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    # initialize the x
    x = np.arange(start_t, end_t, tstep)

    # interp1d t
    t = ts * 50

    # initialize the interp1d x
    x_soft = np.linspace(x.min(), x.max(), t)

    # initialize the interp1d y
    y_soft = np.zeros([n, t])

    # interp1d
    for i in range(n):
        if len(corrs.shape) == 3:
            f = interp1d(x, corrs[i, :, 0], kind='cubic')
            y_soft[i] = f(x_soft)
        if len(corrs.shape) == 2:
            f = interp1d(x, corrs[i, :], kind='cubic')
            y_soft[i] = f(x_soft)

    # get the max value
    vmax = np.max(y_soft)
    # get the min value
    vmin = np.min(y_soft)

    if vmax <= 1/1.1:
        ymax = np.max(y_soft)*1.1
    else:
        ymax = 1

    if vmin >= 0:
        ymin = -0.1
    elif vmin < 0 and vmin > -1/1.1:
        ymin = np.min(y_soft)*1.1
    else:
        ymin = -1

    fig, ax = plt.subplots()

    for i in range(n):

        if labels:
            plt.plot(x_soft, y_soft[i], linewidth=3, label=labels[i])
        else:
            plt.plot(x_soft, y_soft[i], linewidth=3)

    plt.ylim(ymin, ymax)
    plt.ylabel("Similarity", fontsize=20, labelpad=labelpad)
    plt.xlabel("Time (s)", fontsize=20)
    plt.tick_params(labelsize=18)

    if labels:
        plt.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(title, fontsize=title_fontsize)

    plt.show()

    return 0


' a function for plotting the time-by-time Similarities with statistical results'

def plot_tbytsim_withstats(similarities, start_time=0, end_time=1, time_interval=0.01, smooth=True, p=0.05, cbpt=True,
                           clusterp=0.05, stats_time=[0, 1], color='r', xlim=[0, 1], ylim=[-0.1, 0.8],
                           xlabel='Time (s)', ylabel='Representational Similarity', figsize=[6.4, 3.6], x0=0,
                           labelpad=0, ticksize=12, fontsize=16, markersize=2, title=None, title_fontsize=16,
                           avgshow=False):

    """
    Plot the time-by-time Similarities with statistical results

    Parameters
    ----------
    similarities : array
        The Similarities.
        The size of similarities should be [n_subs, n_ts] or [n_subs, n_ts, 2]. n_subs, n_ts represent the number of
        subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    smooth : bool True or False. Default is True.
        Smooth the results or not.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Representational Similarity'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    labelpad : int or float. Default is 0.
        Distance of ylabel from the y-axis.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    """

    if len(np.shape(similarities)) < 2 or len(np.shape(similarities)) > 3:

        return "Invalid input!"

    n = len(np.shape(similarities))

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    if n == 3:
        similarities = similarities[:, :, 0]

    csimilarities = copy.deepcopy(similarities)

    nsubs, nts = np.shape(csimilarities)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:
        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    if smooth is True:
        for sub in range(nsubs):
            for t in range(nts):

                if t<=1:
                    csimilarities[sub, t] = np.average(csimilarities[sub, :t+3])
                if t>1 and t<(nts-2):
                    csimilarities[sub, t] = np.average(csimilarities[sub, t-2:t+3])
                if t>=(nts-2):
                    csimilarities[sub, t] = np.average(csimilarities[sub, t-2:])

    avg = np.average(csimilarities, axis=0)
    err = np.zeros([nts])

    for t in range(nts):
        err[t] = np.std(csimilarities[:, t], ddof=1)/np.sqrt(nsubs)

    if cbpt == True:
        ps_stats = clusterbased_permutation_1d_1samp_1sided(csimilarities[:, stats_time1:stats_time2], level=0,
                                                            p_threshold=p, clusterp_threshold=clusterp)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats
    else:
        ps = np.zeros([nts])
        for t in range(nts):
            ps[t] = ttest_1samp(csimilarities[:, t], 0, alternative="greater")[1]
            if ps[t] < p:
                ps[t] = 1
            else:
                ps[t] = 0

    print('\nSignificant time-windows:')
    for t in range(nts):
        if t == 0 and ps[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == 1 and ps[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == 1 and ps[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    for t in range(nts):
        if ps[t] == 1:
            plt.plot(t*tstep+start_time+0.5*tstep, (ymaxlim-yminlim)*0.95+yminlim, 's',
                     color=color, alpha=0.8, markersize=markersize)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [0]
            ymax = [avg[t]-err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.1)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0))
    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    if avgshow is True:
        plt.plot(x, avg, color=color, alpha=0.95)
    plt.fill_between(x, avg + err, avg - err, facecolor=color, alpha=0.75)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps


' a function for plotting the time-by-time decoding accuracies '

def plot_tbyt_decoding_acc(acc, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                           clusterp=0.05, stats_time=[0, 1], color='r', xlim=[0, 1], ylim=[0.4, 0.8],
                           xlabel='Time (s)', ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, labelpad=0,
                           ticksize=12, fontsize=16, markersize=2, title=None, title_fontsize=16, avgshow=False):

    """
    Plot the time-by-time decoding accuracies

    Parameters
    ----------
    acc : array
        The decoding accuracies.
        The size of acc should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Decoding Accuracy'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    labelpad : int or float. Default is 0.
        Distance of ylabel from the y-axis.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    """

    if len(np.shape(acc)) != 2:

        return "Invalid input!"

    nsubs, nts = np.shape(acc)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:

        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    avg = np.average(acc, axis=0)
    err = np.zeros([nts])
    for t in range(nts):
        err[t] = np.std(acc[:, t], ddof=1) / np.sqrt(nsubs)

    if cbpt == True:

        ps_stats = clusterbased_permutation_1d_1samp_1sided(acc[:, stats_time1:stats_time2], level=chance,
                                                            p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats

    else:
        ps = np.zeros([nts])
        for t in range(nts):
            if t >= stats_time1 and t< stats_time2:
                ps[t] = ttest_1samp(acc[:, t], chance, alternative="greater")[1]
                if ps[t] < p:
                    ps[t] = 1
                else:
                    ps[t] = 0

    print('\nSignificant time-windows:')
    for t in range(nts):
        if t == 0 and ps[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == 1 and ps[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == 1 and ps[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    for t in range(nts):
        if ps[t] == 1:
            plt.plot(t*tstep+start_time+0.5*tstep, (ymaxlim-yminlim)*0.95+yminlim, 's', color=color, alpha=0.8,
                     markersize=markersize)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [chance]
            ymax = [avg[t] - err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.2)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))
    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    if avgshow is True:
        plt.plot(x, avg, color=color, alpha=0.95)
    plt.fill_between(x, avg+err, avg-err, facecolor=color, alpha=0.75)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps

' a function for plotting the differences of time-by-time decoding accuracies between two conditions '

def plot_tbyt_diff_decoding_acc(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                                clusterp=0.05, stats_time=[0, 1], color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, labelpad=0, ticksize=12,
                                fontsize=16, markersize=2, legend_fontsize=14, title=None, title_fontsize=16,
                                avgshow=False):

    """
    Plot the differences of time-by-time decoding accuracies between two conditions

    Parameters
    ----------
    acc1 : array
        The decoding accuracies under condition1.
        The size of acc1 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    acc2 : array
        The decoding accuracies under condition2.
        The size of acc2 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color1 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition1.
    color2 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition2.
    label1 : string-array. Default is 'Condition1'.
        The Label of acc1's condition.
    label2 : string-array. Default is 'Condition2'.
        The Label of acc2's condition.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Decoding Accuracy'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    labelpad : int or float. Default is 0.
        Distance of ylabel from the y-axis.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    legend_fontsize : int or float. Default is 14.
        The fontsize of the legend.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    """

    if len(np.shape(acc1)) != 2 or len(np.shape(acc2)) != 2:

        return "Invalid input!"

    nsubs, nts = np.shape(acc1)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:

        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    avg1 = np.average(acc1, axis=0)
    err1 = np.zeros([nts])
    for t in range(nts):
        err1[t] = np.std(acc1[:, t], ddof=1) / np.sqrt(nsubs)

    avg2 = np.average(acc2, axis=0)
    err2 = np.zeros([nts])
    for t in range(nts):
        err2[t] = np.std(acc2[:, t], ddof=1) / np.sqrt(nsubs)

    if cbpt == True:

        ps1_stats = clusterbased_permutation_1d_1samp_1sided(acc1[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps1 = np.zeros([nts])
        ps1[stats_time1:stats_time2] = ps1_stats
        ps2_stats = clusterbased_permutation_1d_1samp_1sided(acc2[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps2 = np.zeros([nts])
        ps2[stats_time1:stats_time2] = ps2_stats
        ps_stats = clusterbased_permutation_1d_1samp_2sided(acc1[:, stats_time1:stats_time2]-
                                                            acc2[:, stats_time1:stats_time2], level=0, p_threshold=p,
                                                            clusterp_threshold=clusterp, iter=1000)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats

    else:
        ps1 = np.zeros([nts])
        ps2 = np.zeros([nts])
        ps = np.zeros([nts])
        for t in range(nts):
            if t >= stats_time1 and t< stats_time2:
                ps1[t] = ttest_1samp(acc1[:, t], chance, alternative="greater")[1]
                ps2[t] = ttest_1samp(acc2[:, t], chance, alternative="greater")[1]
                if ps1[t] < p:
                    ps1[t] = 1
                else:
                    ps1[t] = 0
                if ps2[t] < p:
                    ps2[t] = 1
                else:
                    ps2[t] = 0
                if ttest_rel(acc1[:, t], acc2[:, t], alternative="greater")[1] < p/2:
                    ps[t] = 1
                elif ttest_rel(acc1[:, t], acc2[:, t], alternative="less")[1] < p/2:
                    ps[t] = -1
                else:
                    ps[t] = 0

    print('\nSignificant time-windows for condition 1:')
    for t in range(nts):
        if t == 0 and ps1[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps1[t] == 1 and ps1[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps1[t] == 1 and ps1[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps1[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 2:')
    for t in range(nts):
        if t == 0 and ps2[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps2[t] == 1 and ps2[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps2[t] == 1 and ps2[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps2[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 1 > condition 2:')
    for t in range(nts):
        if t == 0 and ps[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == 1 and ps2[t - 1] < 1:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == 1 and ps[t + 1] < 1:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 2 > condition 1:')
    for t in range(nts):
        if t == 0 and ps[t] == -1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == -1 and ps2[t - 1] > -1:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == -1 and ps[t + 1] > -1:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == -1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    for t in range(nts):
        if ps1[t] == 1:
            plt.plot(t*tstep+start_time+0.5*tstep, (ymaxlim-yminlim)*0.95+yminlim, 's', color=color1, alpha=0.8,
                     markersize=markersize)
        if ps2[t] == 1:
            plt.plot(t*tstep+start_time+0.5*tstep, (ymaxlim-yminlim)*0.91+yminlim, 's', color=color2, alpha=0.8,
                     markersize=markersize)
        if ps[t] == 1:
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [avg2[t] + err2[t]]
            ymax = [avg1[t] - err1[t]]
            plt.fill_between(xi, ymax, ymin, facecolor="grey", alpha=0.2)
        if ps[t] == -1:
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [avg1[t] + err1[t]]
            ymax = [avg2[t] - err2[t]]
            plt.fill_between(xi, ymax, ymin, facecolor="grey", alpha=0.2)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))
    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    if avgshow is True:
        plt.plot(x, avg1, color=color1, alpha=0.95)
        plt.plot(x, avg2, color=color2, alpha=0.95)
    plt.fill_between(x, avg1+err1, avg1-err1, facecolor=color1, alpha=0.75, label=label1)
    plt.fill_between(x, avg2+err2, avg2-err2, facecolor=color2, alpha=0.75, label=label2)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
    plt.legend()
    ax = plt.gca()
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps1, ps2, ps


' a function for plotting cross-temporal decoding accuracies '

def plot_ct_decoding_acc(acc, start_timex=0, end_timex=1, start_timey=0, end_timey=1, time_intervalx=0.01,
                         time_intervaly=0.01, chance=0.5, p=0.05, cbpt=True, clusterp=0.05, stats_timex=[0, 1],
                         stats_timey=[0, 1], xlim=[0, 1], ylim=[0, 1], clim=[0.4, 0.8], xlabel='Training Time (s)',
                         ylabel='Test Time (s)', clabel='Decoding Accuracy', figsize=[6.4, 4.8], cmap="viridis",
                         ticksize=12, fontsize=16, title=None, title_fontsize=16):

    """
    Plot the cross-temporal decoding accuracies

    Parameters
    ----------
    acc : array
        The decoding accuracies.
        The size of acc should be [n_subs, n_tsx, n_tsy]. n_subs, n_tsx and n_tsy represent the number of subjects,
        the number of training time-points and the number of test time-points.
    start_timex : int or float. Default is 0.
        The training start time.
    end_timex : int or float. Default is 1.
        The training end time.
    start_timey : int or float. Default is 0.
        The test start time.
    end_timey : int or float. Default is 1.
        The test end time.
    time_intervalx : float. Default is 0.01.
        The training time interval between two time samples.
    time_intervaly : float. Default is 0.01.
        The test time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_timex : array or list [stats_timex1, stats_timex2]. Default os [0, 1].
        Trainning time period for statistical analysis.
    stats_timey : array or list [stats_timey1, stats_timey2]. Default os [0, 1].
        Test time period for statistical analysis.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (training time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0, 1].
        The y-axis (test time) view lims.
    clim : array or list [cmin, cmax]. Default is [0.4, 0.8].
        The color-bar (decoding accuracy) view lims.
    xlabel : string. Default is 'Training Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Test Time (s)'.
        The label of y-axis.
    clabel : string. Default is 'Decoding Accuracy'.
        The label of color-bar.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    nsubs, nx, ny = np.shape(acc)
    cminlim = clim[0]
    cmaxlim = clim[1]

    tstepx = float(Decimal((end_timex - start_timex) / nx).quantize(Decimal(str(time_intervalx))))
    tstepy = float(Decimal((end_timey - start_timey) / ny).quantize(Decimal(str(time_intervaly))))

    if tstepx != time_intervalx or tstepy != time_intervaly:
        return "Invalid input!"

    deltax1 = (stats_timex[0] - start_timex) / tstepx - int((stats_timex[0] - start_timex) / tstepx)
    deltax2 = (stats_timex[1] - start_timex) / tstepx - int((stats_timex[1] - start_timex) / tstepx)
    if deltax1 == 0:
        stats_timex1 = int((stats_timex[0] - start_timex) / tstepx)
    else:
        stats_timex1 = int((stats_timex[0] - start_timex) / tstepx) + 1
    if deltax2 == 0:
        stats_timex2 = int((stats_timex[1] - start_timex) / tstepx)
    else:
        stats_timex2 = int((stats_timex[1] - start_timex) / tstepx) + 1

    deltay1 = (stats_timey[0] - start_timey) / tstepy - int((stats_timey[0] - start_timey) / tstepy)
    deltay2 = (stats_timey[1] - start_timey) / tstepy - int((stats_timey[1] - start_timey) / tstepy)
    if deltay1 == 0:
        stats_timey1 = int((stats_timey[0] - start_timey) / tstepy)
    else:
        stats_timey1 = int((stats_timey[0] - start_timey) / tstepy) + 1
    if deltay2 == 0:
        stats_timey2 = int((stats_timey[1] - start_timey) / tstepy)
    else:
        stats_timey2 = int((stats_timey[1] - start_timey) / tstepy) + 1

    if cbpt is True:

        ps_stats = clusterbased_permutation_2d_1samp_1sided(
            acc[:, stats_timex1:stats_timex2, stats_timey1:stats_timey2], level=chance, p_threshold=p,
            clusterp_threshold=clusterp, iter=1000)
        ps = np.zeros([nx, ny])
        ps[stats_timex1:stats_timex2, stats_timey1:stats_timey2] = ps_stats

    else:
        ps = np.zeros([nx, ny])
        for t1 in range(nx):
            for t2 in range(ny):
                if t1 >= stats_timex1 and t1 < stats_timex2 and t2 >= stats_timey1 and t2 < stats_timey2:
                    ps[t1, t2] = ttest_1samp(acc[:, t1, t2], chance, alternative="greater")[1]
                    if ps[t1, t2] < p:
                        ps[t1, t2] = 1
                    else:
                        ps[t1, t2] = 0

    newps = np.zeros([nx + 2, ny + 2])
    newps[1:nx + 1, 1:ny + 1] = ps
    x = np.linspace(start_timex - 0.5 * tstepx, end_timex + 0.5 * tstepx, nx + 2)
    y = np.linspace(start_timey - 0.5 * tstepy, end_timey + 0.5 * tstepy, ny + 2)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, np.transpose(newps, (1, 0)), [0, 1], colors="silver", alpha=0.9, linewidths=3,
                linestyles="dashed")

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    avg = np.average(acc, axis=0)
    avg = np.transpose(avg, (1, 0))
    plt.imshow(avg, extent=(start_timex, end_timex, start_timey, end_timey), cmap=cmap, origin="lower",
               clim=(cminlim, cmaxlim))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=ticksize)
    font = {'size': ticksize+2}
    cb.set_label(clabel, fontdict=font)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps


' a function for plotting the differences of cross-temporal decoding accuracies between two conditions '

def plot_ct_diff_decoding_acc(acc1, acc2, start_timex=0, end_timex=1, start_timey=0, end_timey=1, time_intervalx=0.01,
                              time_intervaly=0.01, p=0.05, cbpt=True, clusterp=0.05, stats_timex=[0, 1],
                              stats_timey=[0, 1], xlim=[0, 1], ylim=[0, 1], clim=[0.4, 0.8], xlabel='Training Time (s)',
                              ylabel='Test Time (s)', clabel='Differences of Decoding Accuracies', figsize=[6.4, 4.8],
                              cmap="viridis", ticksize=12, fontsize=16, title=None, title_fontsize=16):

    """
    Plot the differences of cross-temporal decoding accuracies between two conditions

    Parameters
    ----------
    acc1 : array
        The decoding accuracies under condition1.
        The size of acc should be [n_subs, n_tsx, n_tsy]. n_subs, n_tsx and n_tsy represent the number of subjects,
        the number of training time-points and the number of test time-points.
    acc2 : array
        The decoding accuracies under condition2.
        The size of acc should be [n_subs, n_tsx, n_tsy]. n_subs, n_tsx and n_tsy represent the number of subjects,
        the number of training time-points and the number of test time-points.
    start_timex : int or float. Default is 0.
        The training start time.
    end_timex : int or float. Default is 1.
        The training end time.
    start_timey : int or float. Default is 0.
        The test start time.
    end_timey : int or float. Default is 1.
        The test end time.
    time_intervalx : float. Default is 0.01.
        The training time interval between two time samples.
    time_intervaly : float. Default is 0.01.
        The test time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_timex : array or list [stats_timex1, stats_timex2]. Default os [0, 1].
        Trainning time period for statistical analysis.
    stats_timey : array or list [stats_timey1, stats_timey2]. Default os [0, 1].
        Test time period for statistical analysis.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (training time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0, 1].
        The y-axis (test time) view lims.
    clim : array or list [cmin, cmax]. Default is [0.4, 0.8].
        The color-bar (decoding accuracy) view lims.
    xlabel : string. Default is 'Training Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Test Time (s)'.
        The label of y-axis.
    clabel : string. Default is 'Differences of Decoding Accuracies'.
        The label of color-bar.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    acc = acc1 - acc2
    nsubs, nx, ny = np.shape(acc)
    cminlim = clim[0]
    cmaxlim = clim[1]

    tstepx = float(Decimal((end_timex - start_timex) / nx).quantize(Decimal(str(time_intervalx))))
    tstepy = float(Decimal((end_timey - start_timey) / ny).quantize(Decimal(str(time_intervaly))))

    if tstepx != time_intervalx or tstepy != time_intervaly:
        return "Invalid input!"

    deltax1 = (stats_timex[0] - start_timex) / tstepx - int((stats_timex[0] - start_timex) / tstepx)
    deltax2 = (stats_timex[1] - start_timex) / tstepx - int((stats_timex[1] - start_timex) / tstepx)
    if deltax1 == 0:
        stats_timex1 = int((stats_timex[0] - start_timex) / tstepx)
    else:
        stats_timex1 = int((stats_timex[0] - start_timex) / tstepx) + 1
    if deltax2 == 0:
        stats_timex2 = int((stats_timex[1] - start_timex) / tstepx)
    else:
        stats_timex2 = int((stats_timex[1] - start_timex) / tstepx) + 1

    deltay1 = (stats_timey[0] - start_timey) / tstepy - int((stats_timey[0] - start_timey) / tstepy)
    deltay2 = (stats_timey[1] - start_timey) / tstepy - int((stats_timey[1] - start_timey) / tstepy)
    if deltay1 == 0:
        stats_timey1 = int((stats_timey[0] - start_timey) / tstepy)
    else:
        stats_timey1 = int((stats_timey[0] - start_timey) / tstepy) + 1
    if deltay2 == 0:
        stats_timey2 = int((stats_timey[1] - start_timey) / tstepy)
    else:
        stats_timey2 = int((stats_timey[1] - start_timey) / tstepy) + 1

    if cbpt is True:

        ps_stats = clusterbased_permutation_2d_2sided(acc1[:, stats_timex1:stats_timex2, stats_timey1:stats_timey2],
                                                      acc2[:, stats_timex1:stats_timex2, stats_timey1:stats_timey2],
                                                      p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps = np.zeros([nx, ny])
        ps[stats_timex1:stats_timex2, stats_timey1:stats_timey2] = ps_stats

    else:
        ps = np.zeros([nx, ny])
        for t1 in range(nx):
            for t2 in range(ny):
                if t1 >= stats_timex1 and t1 < stats_timex2 and t2 >= stats_timey1 and t2 < stats_timey2:
                    if ttest_1samp(acc[:, t1, t2], 0, alternative="greater")[1] < p/2:
                        ps[t1, t2] = 1
                    elif ttest_1samp(acc[:, t1, t2], 0, alternative="less")[1] < p/2:
                        ps[t1, t2] = -1
                    else:
                        ps[t1, t2] = 0

    newps = np.zeros([nx + 2, ny + 2])
    newps[1:nx + 1, 1:ny + 1] = ps
    x = np.linspace(start_timex - 0.5 * tstepx, end_timex + 0.5 * tstepx, nx + 2)
    y = np.linspace(start_timey - 0.5 * tstepy, end_timey + 0.5 * tstepy, ny + 2)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, np.transpose(newps, (1, 0)), (-0.5, 0.5), colors="silver", alpha=0.9, linewidths=3,
                linestyles="dashed")

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    avg = np.average(acc, axis=0)
    avg = np.transpose(avg, (1, 0))
    plt.imshow(avg, extent=(start_timex, end_timex, start_timey, end_timey), cmap=cmap, origin="lower",
               clim=(cminlim, cmaxlim))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=ticksize)
    font = {'size': ticksize+2}
    cb.set_label(clabel, fontdict=font)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps


' a function for plotting the hotmap of correlations coefficients for channels/regions by time sequence '

def plot_corrs_hotmap(corrs, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], smooth=False, figsize=None, cmap=None,
                      title=None, title_fontsize=16):

    """
    plot the hotmap of correlation coefficients for channels/regions by time sequence

    corrs : array
        The correlation coefficients time-by-time.
        The shape of corrs must be [n_chls, ts, 2] or [n_chls, ts]. n_chls represents the number of channels or
        regions. ts represents the number of time-points. If shape of corrs is [n_chls, ts 2], each time-point
        of each channel/region contains a r-value and a p-value. If shape is [n_chls, ts], only r-values.
    chllabel : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    smooth : bool True or False. Default is False.
        Smooth the results or not.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'inferno'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(corrs)) < 2 or len(np.shape(corrs)) > 3:

        return "Invalid input!"

    # get the number of channels
    nchls = corrs.shape[0]

    # get the number of time-points
    ts = corrs.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    # initialize the x
    x = np.arange(start_t, end_t, tstep)

    # set labels of the channels
    if chllabels == None:

        chllabels = []
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

    # smooth the results
    if smooth == True:

        t = ts * 50

        x_soft = np.linspace(x.min(), x.max(), t)
        y_soft = np.zeros([nchls, t])

        samplerate = int(1 / tstep) * 50
        b, a = signal.butter(4, 2*30/samplerate, 'lowpass')

        for i in range(nchls):

            if len(corrs.shape) == 3:
                f = interp1d(x, corrs[i, :, 0], kind='cubic')
                y_soft[i] = f(x_soft)
            elif len(corrs.shape) == 2:
                f = interp1d(x, corrs[i, :], kind='cubic')
                y_soft[i] = f(x_soft)
                y_soft[i] = signal.filtfilt(b, a, y_soft[i])

        rlts = y_soft

    if smooth == False:

        if len(corrs.shape) == 3:
            rlts = corrs[:, :, 0]
        elif len(corrs.shape) == 2:
            rlts = corrs

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts * tstep * (size[0] - 2) + 2
        size_y = nchls * 0.2 * (size[1] - 1.5) + 1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

    delta = (size_y * 3) / (size_x * 4)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*0.16*delta), clim=(limmin, limmax), origin='lower', cmap='inferno')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls * 0.16*delta), clim=(limmin, limmax), origin='lower', cmap=cmap)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Similarity", fontdict=font)

    xi = []

    for i in range(nchls):
        xi.append(0.16*delta*i + 0.08*delta)

    yi = chllabels

    plt.tick_params(labelsize=18)
    plt.yticks(xi, yi, fontsize=18)
    plt.ylabel("Channel", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return 0


' a function for plotting the hotmap of correlations coefficients for channels/regions by time sequence with the significant outline '

def plot_corrs_hotmap_withstats(corrs, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], p=0.05, cbpt=False,
                                clusterp=0.05, stats_time=[0, 1], smooth=False, xlabel='Time (s)', ylabel='Channel',
                                clabel='Similarity', ticksize=18, figsize=None, cmap=None, title=None,
                                title_fontsize=16):

    """
    plot the hotmap of correlation coefficients for channels/regions by time sequence with the significant outline

    corrs : array
        The correlation coefficients time-by-time.
        The shape of corrs must be [n_subs, n_chls, ts, 2] or [n_subs, n_chls, ts]. n_subs represents the number of
        subjects. n_chls represents the number of channels or regions. ts represents the number of time-points. If shape
        of corrs is [n_chls, ts 2], each time-point of each channel/region contains a r-value and a p-value. If shape is
        [n_chls, ts], only r-values.
    chllabels : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    p: float. Default is 0.05.
        The p threshold for outline.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        The time period for statistical analysis.
    smooth : bool True or False. Default is False.
        Smooth the results or not.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Channel'.
        The label of y-axis.
    clabel : string. Default is 'Similarity'.
        The label of color-bar.
    ticksize : int or float. Default is 18.
        The size of the ticks.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the colormap will be 'inferno'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(corrs)) < 3 or len(np.shape(corrs)) > 4:

        return "Invalid input!"

    # get the number of channels
    nchls = corrs.shape[1]

    # get the number of time-points
    nts = corrs.shape[2]

    # get the start time and the time step
    start_time = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_time = start_time + nts * tstep

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    # set labels of the channels
    if chllabels == None:

        chllabels = []
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

    if len(corrs.shape) == 4:
        rlts = corrs[:, :, :, 0]
    elif len(corrs.shape) == 3:
        rlts = corrs

    # smooth the results
    if smooth == True:

        for chl in range(nchls):
            rlts[:, chl] = smooth_1d(rlts[:, chl])

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = nts * tstep * (size[0] - 2) + 2
        size_y = nchls * 0.2 * (size[1] - 1.5) + 1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

    delta = (size_y * 3) / (size_x * 4)

    avg = np.average(rlts, axis=0)

    ps = np.zeros([nchls, nts])

    if cbpt == True:

        for chl in range(nchls):
            ps_stats = clusterbased_permutation_1d_1samp_2sided(rlts[:, chl, stats_time1:stats_time2], 0, p_threshold=p,
                                                                clusterp_threshold=clusterp, iter=1000)
            ps[chl, stats_time1:stats_time2] = ps_stats

    else:
        for chl in range(nchls):
            for t in range(nts):
                if t >= stats_time1 and t < stats_time2:
                    ps[chl, t] = ttest_1samp(rlts[:, chl, t], 0)[1]
                    if ps[chl, t] < p and avg[chl, t] > 0:
                        ps[chl, t] = 1
                    elif ps[chl, t] < p and avg[chl, t] < 0:
                        ps[chl, t] = -1
                    else:
                        ps[chl, t] = 0

    newps = np.zeros([nchls + 2, nts + 2])
    newps[1:nchls + 1, 1:nts + 1] = ps

    x = np.linspace(start_time - 0.5 * tstep, end_time + 0.5 * tstep, nts + 2)
    y = np.linspace(-0.08*delta, 0.16*delta * nchls + 0.08*delta, nchls + 2)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, newps, [0.5], linewidths=2, linestyles="dashed")
    plt.contour(X, Y, newps, [-0.5], linewidths=2, linestyles="dashed")

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(avg, extent=(start_time, end_time, 0, nchls*delta*0.16), clim=(limmin, limmax), origin='lower', cmap='inferno')
    else:
        plt.imshow(avg, extent=(start_time, end_time, 0, nchls*delta * 0.16), clim=(limmin, limmax), origin='lower', cmap=cmap)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=ticksize-2)
    font = {'size': ticksize}
    cb.set_label(clabel, fontdict=font)

    xi = []

    for i in range(nchls):
        xi.append(0.16*delta*i + 0.08*delta)

    yi = chllabels

    plt.tick_params(labelsize=ticksize)
    plt.yticks(xi, yi, fontsize=ticksize)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps


' a function for plotting the hotmap of neural pattern similarities for channels/regions by time sequence '

def plot_nps_hotmap(similarities, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], abs=False, smooth=False, figsize=None,
                    cmap=None, title=None, title_fontsize=16):

    """
    plot the hotmap of neural pattern similarities for channels/regions by time sequence

    similarities : array
        The neural pattern similarities time-by-time.
        The shape of similarities must be [n_chls, ts]. n_chls represents the number of channels or regions.
        ts represents the number of time-points.
    chllabel : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    abs : boolean True or False. Default is False.
        Change the similarities into absolute values or not.
    smooth : boolean True or False. Default is False.
        Smooth the results or not.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'viridis'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(similarities)) != 2:

        return "Invalid input!"

    # absolute value
    if abs == True:
        similarities = np.abs(similarities)

    # get the number of channels
    nchls = similarities.shape[0]

    # get the number of time-points
    ts = similarities.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    # initialize the x
    x = np.arange(start_t, end_t, tstep)

    # set labels of the channels
    if chllabels == None:

        chllabels = []
        for i in range(nchls):

            if i % 10 == 0 and i != 10:
                newlabel = str(i + 1) + "st"
            elif i % 10 == 1 and i != 11:
                newlabel = str(i + 1) + "nd"
            elif i % 10 == 2 and i != 12:
                newlabel = str(i + 1) + "rd"
            else:
                newlabel = str(i + 1) + "th"
            chllabels.append(newlabel)

    if smooth == True:

        t = ts * 50

        x_soft = np.linspace(x.min(), x.max(), t)
        y_soft = np.zeros([nchls, t])

        samplerate = int(1 / tstep) * 50
        b, a = signal.butter(4, 2*30/samplerate, 'lowpass')

        for i in range(nchls):
            f = interp1d(x, similarities[i, :], kind='cubic')
            y_soft[i] = f(x_soft)
            y_soft[i] = signal.filtfilt(b, a, y_soft[i])

        rlts = y_soft

    if smooth == False:
        rlts = similarities

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts * tstep * (size[0] - 2) + 2
        size_y = nchls * 0.2 * (size[1] - 1.5) + 1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

    delta = (size_y * 3) / (size_x * 4)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*delta*0.16), clim=(limmin, limmax), origin='lower')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*delta*0.16), clim=(limmin, limmax), origin='lower', cmap=cmap)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Similarity", fontdict=font)

    xi = []

    for i in range(nchls):
        xi.append(0.16*delta*i + 0.08*delta)

    yi = chllabels

    plt.tick_params(labelsize=18)
    plt.yticks(xi, yi, fontsize=18)
    plt.ylabel("Channel", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return 0


' a function for plotting the hotmap of statistical results for channels/regions by time sequence '

def plot_t_hotmap_withstats(results, chllabels=None, time_unit=[0, 0.1], lim=[-7, 7], p=0.05, cbpt=False,
                            clusterp=0.05, stats_time=[0, 1], smooth=False, xlabel='Time (s)', ylabel='Channel',
                            clabel='t', ticksize=18, figsize=None, cmap=None, title=None, title_fontsize=16):

    """
    plot the hotmap of statistical results for channels/regions by time sequence

    results : array
        The results.
        The shape of results must be [n_subs, n_chls, ts, 2] or [n_subs, n_chls, ts]. n_subs represents the number of
        subjects. n_chls represents the number of channels or regions. ts represents the number of time-points. If shape
        of corrs is [n_chls, ts 2], each time-point of each channel/region contains a r-value and a p-value. If shape is
        [n_chls, ts], only r-values.
    chllabels : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    p: float. Default is 0.05.
        The p threshold for outline.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        The time period for statistical analysis.
    smooth : bool True or False. Default is False.
        Smooth the results or not.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Channel'.
        The label of y-axis.
    clabel : string. Default is 'Similarity'.
        The label of color-bar.
    ticksize : int or float. Default is 18.
        The size of the ticks.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'bwr'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """

    if len(np.shape(results)) < 3 or len(np.shape(results)) > 4:
        return "Invalid input!"

    # get the number of channels
    nchls = results.shape[1]

    # get the number of time-points
    nts = results.shape[2]

    # get the start time and the time step
    start_time = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_time = start_time + nts * tstep

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    # set labels of the channels
    if chllabels == None:

        chllabels = []
        for i in range(nchls):

            if i % 10 == 0 and i != 10:
                newlabel = str(i + 1) + "st"
            elif i % 10 == 1 and i != 11:
                newlabel = str(i + 1) + "nd"
            elif i % 10 == 2 and i != 12:
                newlabel = str(i + 1) + "rd"
            else:
                newlabel = str(i + 1) + "th"

            chllabels.append(newlabel)

    if len(results.shape) == 4:
        rlts = results[:, :, :, 0]
    elif len(results.shape) == 3:
        rlts = results

    # smooth the results
    if smooth == True:

        for chl in range(nchls):
            rlts[:, chl] = smooth_1d(rlts[:, chl])

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = nts * tstep * (size[0] - 2) + 2
        size_y = nchls * 0.2 * (size[1] - 1.5) + 1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

    delta = (size_y * 3) / (size_x * 4)

    ts = ttest_1samp(rlts, 0, axis=0)[:, :, 0]

    ps = np.zeros([nchls, nts])
    if cbpt == True:

        for chl in range(nchls):
            ps_stats = clusterbased_permutation_1d_1samp_2sided(rlts[:, chl, stats_time1:stats_time2], 0, p_threshold=p,
                                                                clusterp_threshold=clusterp, iter=1000)
            ps[chl, stats_time1:stats_time2] = ps_stats

    else:
        for chl in range(nchls):
            for t in range(nts):
                if t >= stats_time1 and t < stats_time2:
                    ps[chl, t] = ttest_1samp(rlts[:, chl, t], 0)[1]
                    if ps[chl, t] < p and ts[chl, t] > 0:
                        ps[chl, t] = 1
                    elif ps[chl, t] < p and ts[chl, t] < 0:
                        ps[chl, t] = -1
                    else:
                        ps[chl, t] = 0

    newps = np.zeros([nchls + 2, nts + 2])
    newps[1:nchls + 1, 1:nts + 1] = ps

    x = np.linspace(start_time - 0.5 * tstep, end_time + 0.5 * tstep, nts + 2)
    y = np.linspace(-0.08*delta, 0.16*delta * nchls + 0.08*delta, nchls + 2)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, newps, [0.5], linewidths=2, linestyles="dashed")
    plt.contour(X, Y, newps, [-0.5], linewidths=2, linestyles="dashed")

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(ts, extent=(start_time, end_time, 0, nchls * 0.16*delta), clim=(limmin, limmax), origin='lower',
                   cmap='bwr')
    else:
        plt.imshow(ts, extent=(start_time, end_time, 0, nchls * 0.16*delta), clim=(limmin, limmax), origin='lower',
                   cmap=cmap)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=ticksize-2)
    font = {'size': ticksize}
    cb.set_label(clabel, fontdict=font)

    xi = []

    for i in range(nchls):
        xi.append(0.16*delta * i + 0.08*delta)

    yi = chllabels

    plt.tick_params(labelsize=ticksize)
    plt.yticks(xi, yi, fontsize=ticksize)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)

    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps


' a function for plotting the RSA-result regions by 3 cuts (frontal, axial & lateral) '

def plot_brainrsa_regions(img, threshold=None, background=get_bg_ch2(), type='r'):

    """
    Plot the RSA-result regions by 3 cuts (frontal, axial & lateral)

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    background : Niimg-like object or string. Default is stuff.get_bg_ch2()
        The background image that the RSA results will be plotted on top of.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_fdata()

    if (imgarray == np.nan).all() == True:
        print("No Valid Results")

    else:
        if threshold != None:

            imgarray = nib.load(img).get_fdata()
            affine = get_affine(img)

            imgarray = correct_by_threshold(imgarray, threshold)

            img = nib.Nifti1Image(imgarray, affine)

        if type == 'r':
            plotting.plot_roi(roi_img=img, bg_img=background, threshold=0, vmin=0.1, vmax=1,
                          title="Similarity", resampling_interpolation="continuous")
        if type == 't':
            plotting.plot_roi(roi_img=img, bg_img=background, threshold=0, vmin=-7, vmax=7,
                              title="Similarity", resampling_interpolation="continuous")

        plt.show()

    return 0


' a function for plotting the RSA-result by different cuts '

def plot_brainrsa_montage(img, threshold=None, slice=[6, 6, 6], background=get_bg_ch2bet(), type='r'):

    """
    Plot the RSA-result by different cuts

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    slice : array
        The point where the cut is performed.
        If slice=[slice_x, slice_y, slice_z], slice_x, slice_y, slice_z represent the coordinates of each cut in the x,
        y, z direction. If slice=[[slice_x1, slice_x2], [slice_y1, slice_y2], [slice_z1, slice_z2]], slice_x1 & slice_x2
        represent the coordinates of each cut in the x direction, slice_y1 & slice_y2 represent the coordinates of each
        cut in the y direction, slice_z1 & slice_z2 represent the coordinates of each cut in the z direction.
    background : Niimg-like object or string. Default is stuff.get_bg_ch2bet()
        The background image that the RSA results will be plotted on top of.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_fdata()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:

        if threshold != None:
            imgarray = nib.load(img).get_fdata()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        slice_x = slice[0]
        slice_y = slice[1]
        slice_z = slice[2]

        if type == 'r':
            vmax = 1
        if type == 't':
            vmax = 7

        if slice_x != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='x', cut_coords=slice_x,
                                title="Similarity -sagittal", draw_cross=True, vmax=vmax)

        if slice_y != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='y', cut_coords=slice_y,
                                title="Similarity -coronal", draw_cross=True, vmax=vmax)

        if slice_z != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='z', cut_coords=slice_z,
                                title="Similarity -axial", draw_cross=True, vmax=vmax)

        plt.show()

    return 0


' a function for plotting the 2-D projection of the RSA-result '

def plot_brainrsa_glass(img, threshold=None, type='r'):

    """
    Plot the 2-D projection of the RSA-result

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_fdata()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:
        if threshold != None:

            imgarray = nib.load(img).get_fdata()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        if type == 'r':
            plotting.plot_glass_brain(img, colorbar=True, title="Similarity", black_bg=True, draw_cross=True, vmax=1)
        if type == 't':
            plotting.plot_glass_brain(img, colorbar=True, title="Similarity", black_bg=True, draw_cross=True, vmax=7)

        plt.show()

    return 0


' a function for plotting the RSA-result into a brain surface '

def plot_brainrsa_surface(img, threshold=None, type='r'):

    """
    Plot the RSA-result into a brain surface

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_fdata()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:

        if threshold != None:

            imgarray = nib.load(img).get_fdata()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
        texture_left = surface.vol_to_surf(img, fsaverage.pial_left)
        texture_right = surface.vol_to_surf(img, fsaverage.pial_right)

        # type='r'
        if type == 'r':
            plotting.plot_surf_stat_map(fsaverage.pial_left, texture_left, hemi='left', threshold=0.1,
                                        bg_map=fsaverage.sulc_right, colorbar=False, vmax=0.8, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_right, texture_right, hemi='right', threshold=0.1,
                                        bg_map=fsaverage.sulc_right, colorbar=True, vmax=0.8, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_right, texture_left, hemi='left', threshold=0.1,
                                        bg_map=fsaverage.sulc_right, colorbar=False, vmax=0.8, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_left, texture_right, hemi='right', threshold=0.1,
                                        bg_map=fsaverage.sulc_right, colorbar=True, vmax=0.8, darkness=0.7)

            plt.show()

        # type='t'
        if type == 't':
            plotting.plot_surf_stat_map(fsaverage.pial_left, texture_left, hemi='left', threshold=0.8,
                                        bg_map=fsaverage.sulc_right, colorbar=False, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_right, texture_right, hemi='right', threshold=0.8,
                                        bg_map=fsaverage.sulc_right, colorbar=True, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_right, texture_left, hemi='left', threshold=0.8,
                                        bg_map=fsaverage.sulc_right, colorbar=False, darkness=0.7)

            plotting.plot_surf_stat_map(fsaverage.pial_left, texture_right, hemi='right', threshold=0.8,
                                        bg_map=fsaverage.sulc_right, colorbar=True, darkness=0.7)

            plt.show()

    return 0



' a function for plotting the RSA-result by a set of images '

def plot_brainrsa_rlts(img, threshold=None, slice=[6, 6, 6], background=None, type='r'):

    """
    Plot the RSA-result by a set of images

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    background : Niimg-like object or string. Default is None.
        The background image that the RSA results will be plotted on top of.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_fdata()

    if (imgarray == np.nan).all() == True:
        print("No Valid Results")
    else:

        if threshold != None:

            imgarray = nib.load(img).get_fdata()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        if background == None:

            plot_brainrsa_regions(img, threshold=threshold, type=type)

            plot_brainrsa_montage(img, threshold=threshold, slice=slice, type=type)

            plot_brainrsa_glass(img, threshold=threshold, type=type)

            plot_brainrsa_surface(img, threshold=threshold, type=type)

        else:

            plot_brainrsa_regions(img, threshold=threshold, background=background, type=type)

            plot_brainrsa_montage(img, threshold=threshold, slice=slice, background=background, type=type)

            plot_brainrsa_surface(img, threshold=threshold, type=type)

    return 0