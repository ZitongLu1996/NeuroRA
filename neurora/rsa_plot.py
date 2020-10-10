# -*- coding: utf-8 -*-

' a module for plotting the NeuroRA results '

__author__ = 'Zitong Lu'

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from nilearn import plotting, datasets, surface
import nibabel as nib
from neurora.stuff import get_affine, permutation_test, correct_by_threshold, get_bg_ch2, get_bg_ch2bet


' a function for plotting the RDM '

def plot_rdm(rdm, percentile=False, rescale=False, lim=[0, 1], conditions=None, con_fontsize=12, cmap=None):

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
    """

    # get the number of conditions
    cons = rdm.shape[0]

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    # determine if it's a square
    a, b = np.shape(rdm)
    if a != b:
        return None

    if percentile == True:

        v = np.zeros([cons * cons, 2], dtype=np.float)
        for i in range(cons):
            for j in range(cons):
                v[i * cons + j, 0] = rdm[i, j]

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
                rdm[i, j] = v[i * cons + j, 0]

        if cmap == None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 100))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(0, 100))

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
                        rdm[i, j] = float((rdm[i, j] - minvalue) / (maxvalue - minvalue))

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    else:

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

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

    plt.show()


' a function for plotting the RDM with values '

def plot_rdm_withvalue(rdm, lim=[0, 1], value_fontsize=10, conditions=None, con_fontsize=12, cmap=None):

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
    """

    # get the number of conditions
    cons = rdm.shape[0]

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    # determine if it's a square
    a, b = np.shape(rdm)
    if a != b:
        return None

    # plot the RDM
    min = lim[0]
    max = lim[1]
    if cmap == None:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.Greens, clim=(min, max))
    else:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Dissimilarity", fontdict=font)

    # add values
    step = float(1 / cons)
    for i in range(cons):
        for j in range(cons):
            print(i, j)
            text = plt.text(i * step + 0.5 * step, 1 - j * step - 0.5 * step, float('%.4f' % rdm[i, j]),
                            ha="center", va="center", color="blue", fontsize=value_fontsize)

    if conditions != None:
        print("1")
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")



    plt.show()


' a function for plotting the correlation coefficients by time sequence '

def plot_corrs_by_time(corrs, labels=None, time_unit=[0, 0.1]):

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
    """

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
    plt.ylabel("Similarity", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    plt.tick_params(labelsize=18)

    if labels:
        plt.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()


def plot_tbytsim_withstats(Similarities, start_time=0, end_time=1, color='r', lim=[-0.1, 0.8]):

    """
    Plot the time-by-time Similarities averaging all subjects

    Parameters
    ----------
    Similarities : array
        The Similarities.
        The size of Similarities should be [n_subs, n_ts] or [n_subs, n_ts, 2]. n_subs, n_ts represent the number of
        subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    lim : array or list [min, max]. Default is [-0.1, 0.8].
        The corrs view lims.
    """

    n = len(np.shape(Similarities))

    minlim = lim[0]
    maxlim = lim[1]

    if n == 3:
        Similarities = Similarities[:, :, 0]

    nsubs = np.shape(Similarities)[0]
    nts = np.shape(Similarities)[1]

    tstep = float((end_time-start_time)/nts)

    for sub in range(nsubs):
        for t in range(nts):

            if t<=1:
                Similarities[sub, t] = np.average(Similarities[sub, :t+3])
            if t>1 and t<(nts-2):
                Similarities[sub, t] = np.average(Similarities[sub, t-2:t+3])
            if t>=(nts-2):
                Similarities[sub, t] = np.average(Similarities[sub, t-2:])

    avg = np.average(Similarities, axis=0)
    err = np.zeros([nts], dtype=np.float)

    for t in range(nts):
        err[t] = np.std(Similarities[:, t], ddof=1)/np.sqrt(nsubs)

    ps = np.zeros([nts], dtype=np.float)
    chance = np.full([nsubs], 0)

    for t in range(nts):
        ps[t] = permutation_test(Similarities[:, t], chance)
        if ps[t] < 0.05 and avg[t] > 0:
            plt.plot(t*tstep+start_time, maxlim*0.9, 's', color=color, alpha=1)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [0]
            ymax = [avg[t]-err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.1)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0))

    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    plt.fill_between(x, avg + err, avg - err, facecolor=color, alpha=0.8)
    plt.ylim(minlim, maxlim)
    plt.xlim(start_time, end_time)
    plt.tick_params(labelsize=12)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Representational Similarity", fontsize=16)
    plt.show()


' a function for plotting the hotmap of correlations coefficients for channels/regions by time sequence '

def plot_corrs_hotmap(corrs, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], smooth=False, figsize=None, cmap=None):

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
    """

    # get the number of channels
    nchls = corrs.shape[0]

    # get the number of time-points
    ts = corrs.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    print(start_t, tstep, end_t)

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

    print(rlts.shape)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*0.16), clim=(limmin, limmax), origin='low', cmap='inferno')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls * 0.16), clim=(limmin, limmax), origin='low', cmap=cmap)

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts*tstep*(size[0]-2)+2
        size_y = nchls*0.2*(size[1]-1.5)+1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

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


' a function for plotting the hotmap of correlations coefficients for channels/regions by time sequence with the significant outline '

def plot_corrs_hotmap_stats(corrs, stats, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], p_threshold=0.05, time_threshold=5, smooth=False, figsize=None, cmap=None):

    """
    plot the hotmap of correlation coefficients for channels/regions by time sequence with the significant outline

    corrs : array
        The correlation coefficients time-by-time.
        The shape of corrs must be [n_chls, ts, 2] or [n_chls, ts]. n_chls represents the number of channels or
        regions. ts represents the number of time-points. If shape of corrs is [n_chls, ts 2], each time-point
        of each channel/region contains a r-value and a p-value. If shape is [n_chls, ts], only r-values.
    stats : array
        The statistical results.
        The shape of stats must be [n_chls, ts, 2]. n_chls represents the number of channels or regions.
        ts represents the number of time-points. 2 represents a t-value and a p-value.
    chllabel : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    p_threshold: float. Default is 0.05.
        The p threshold for outline.
    time_threshold: int. Default is 5.
        The time threshold for outline.
        If threshold=5, the time threshold is a window of 5 time-points for each channel/region.
    smooth : bool True or False. Default is False.
        Smooth the results or not.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'inferno'.
    """

    # get the number of channels
    nchls = corrs.shape[0]

    # get the number of time-points
    ts = corrs.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    print(start_t, tstep, end_t)

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

    print(rlts.shape)

    statscopy = stats.copy()

    ps = statscopy[:, :, 1]
    tvalues = statscopy[:, :, 0]

    for i in range(nchls):
        for j in range(ts):

            if ps[i, j] < p_threshold and tvalues[i, j] > 0:
                ps[i, j] = 1
            elif ps[i, j] < p_threshold and tvalues[i, j] < 0:
                ps[i, j] = -1
            else:
                ps[i, j] = 0

    for i in range(nchls):
        pid = set(())
        for j in range(ts):
            if ps[i, j] != 0:
                pid.add(j)
        pid_list = list(pid)
        pid_list.sort()
        pid_set = set()
        for j in pid_list:
            index = 0
            for k in range(time_threshold):
                if j + k in pid_list:
                    index = index
                else:
                    index = index + 1
            if index == 0:
                for k in range(time_threshold):
                    pid_set.add(j + k)
        pid_list = list(pid_set)
        pid_list.sort()
        for j in range(ts):
            index = j in pid_list
            if index is False:
                ps[i, j] = 0

    newps = np.zeros([nchls + 2, ts + 2], dtype=np.float)
    newps[1:nchls + 1, 1:ts + 1] = ps

    x = np.linspace(start_t - 0.5 * tstep, end_t + 0.5 * tstep, ts + 2)
    y = np.linspace(-0.08, 0.16 * nchls + 0.08, nchls + 2)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, newps, (-0.5, 0.5), linewidths=3)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*0.16), clim=(limmin, limmax), origin='low', cmap='inferno')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls * 0.16), clim=(limmin, limmax), origin='low', cmap=cmap)

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts*tstep*(size[0]-2)+2
        size_y = nchls*0.2*(size[1]-1.5)+1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

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


' a function for plotting the hotmap of neural pattern similarities for channels/regions by time sequence '

def plot_nps_hotmap(similarities, chllabels=None, time_unit=[0, 0.1], lim=[0, 1], abs=False, smooth=False, figsize=None, cmap=None):

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
    """

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

    print(start_t, tstep, end_t)

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

    print(rlts.shape)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*0.16), clim=(limmin, limmax), origin='low')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls * 0.16), clim=(limmin, limmax), origin='low', cmap=cmap)

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts*tstep*(size[0]-2)+2
        size_y = nchls*0.2*(size[1]-1.5)+1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

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


' a function for plotting the hotmap of statistical results for channels/regions by time sequence '

def plot_stats_hotmap(stats, chllabels=None, time_unit=[0, 0.1], lim=[-7, 7], smooth=False, figsize=None, cmap=None, outline=False, p_threshold=0.05, time_threshold=5):

    """
    plot the hotmap of statistical results for channels/regions by time sequence

    stats : array
        The statistical results.
        The shape of stats must be [n_chls, ts, 2]. n_chls represents the number of channels or regions.
        ts represents the number of time-points. 2 represents a t-value and a p-value.
    chllabel : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [-7, -7].
        The corrs view lims.
    smooth : boolean True or False. Default is False.
        Smooth the results or not.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'bwr'.
    outline : bool True or False. Default is False.
        Outline the significant areas or not.
    p_threshold: float. Default is 0.05.
        The p threshold for outline.
    time_threshold: int. Default is 5.
        The time threshold for outline.
        If threshold=5, the time threshold is a window of 5 time-points for each channel/region.
    """

    statscopy = stats.copy()

    # get the number of channels
    nchls = statscopy.shape[0]

    # get the number of time-points
    ts = statscopy.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    print(start_t, tstep, end_t)

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
            f = interp1d(x, statscopy[i, :, 0], kind='cubic')
            y_soft[i] = f(x_soft)
            y_soft[i] = signal.filtfilt(b, a, y_soft[i])

        rlts = y_soft

    if smooth == False:
        rlts = statscopy[:, :, 0]

    print(rlts.shape)

    # get min of lims & max of lims
    limmin = lim[0]
    limmax = lim[1]

    if outline == True:
        ps = statscopy[:, :, 1]
        tvalues = statscopy[:, :, 0]

        for i in range(nchls):
            for j in range(ts):

                if ps[i, j] < p_threshold and tvalues[i, j] > 0:
                    ps[i, j] = 1
                elif ps[i, j] < p_threshold and tvalues[i, j] < 0:
                    ps[i, j] = -1
                else:
                    ps[i, j] = 0

        for i in range(nchls):
            pid = set(())
            for j in range(ts):
                if ps[i, j] != 0:
                    pid.add(j)
            pid_list = list(pid)
            pid_list.sort()
            pid_set = set()
            for j in pid_list:
                index = 0
                for k in range(time_threshold):
                    if j+k in pid_list:
                        index = index
                    else:
                        index = index + 1
                if index == 0:
                    for k in range(time_threshold):
                        pid_set.add(j+k)
            pid_list = list(pid_set)
            pid_list.sort()
            for j in range(ts):
                index = j in pid_list
                if index is False:
                    ps[i, j] = 0

        newps = np.zeros([nchls+2, ts+2], dtype=np.float)
        newps[1:nchls+1, 1:ts+1] = ps

        x = np.linspace(start_t-0.5*tstep, end_t+0.5*tstep, ts+2)
        y = np.linspace(-0.08, 0.16*nchls+0.08, nchls+2)
        X, Y = np.meshgrid(x, y)
        plt.contour(X, Y, newps, (-0.5, 0.5), linewidths=3)

    fig = plt.gcf()
    size = fig.get_size_inches()

    if figsize == None:
        size_x = ts*tstep*(size[0]-2)+2
        size_y = nchls*0.2*(size[1]-1.5)+1.5
    else:
        size_x = figsize[0]
        size_y = figsize[1]

    fig.set_size_inches(size_x, size_y)

    if cmap == None:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls*0.16), clim=(limmin, limmax), origin='low')
    else:
        plt.imshow(rlts, extent=(start_t, end_t, 0, nchls * 0.16), clim=(limmin, limmax), origin='low', cmap=cmap)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("t", fontdict=font)

    xi = []

    for i in range(nchls):
        xi.append(0.16*i + 0.08)

    yi = chllabels

    plt.tick_params(labelsize=18)
    plt.yticks(xi, yi, fontsize=18)

    plt.ylabel("Channel", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)

    plt.show()


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

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:
        print("No Valid Results")

    else:
        if threshold != None:

            imgarray = nib.load(img).get_data()
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

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:

        if threshold != None:
            imgarray = nib.load(img).get_data()
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

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:
        if threshold != None:

            imgarray = nib.load(img).get_data()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        if type == 'r':
            plotting.plot_glass_brain(img, colorbar=True, title="Similarity", black_bg=True, draw_cross=True, vmax=1)
        if type == 't':
            plotting.plot_glass_brain(img, colorbar=True, title="Similarity", black_bg=True, draw_cross=True, vmax=7)

        plt.show()


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

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:

        if threshold != None:

            imgarray = nib.load(img).get_data()
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

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:
        print("No Valid Results")
    else:

        if threshold != None:

            imgarray = nib.load(img).get_data()
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
