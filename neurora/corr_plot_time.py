# -*- coding: utf-8 -*-

' a module for plot the correlation coefficients by time sequence '

__author__ = 'Zitong Lu'

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def plot_corrs_bytime(corrs, time_unit=[0, 1]):
    # corrs represent the correlation coefficients point-by-point, its shape : [ts]
    # time_unit=[start_t, t_step]

    ts = len(corrs[:, 0])

    start_t = time_unit[0]
    t_step = time_unit[1]

    end_t = start_t + ts * t_step

    x = np.arange(start_t, end_t, t_step)

    print(x)
    print(corrs)

    x_soft = np.linspace(x.min(), x.max(), 500)

    print(x_soft)

    y_soft = spline(x, corrs[:, 0], x_soft)

    fig, ax = plt.subplots()

    plt.plot(x_soft, y_soft, color="g", linewidth=4)

    plt.ylim(0, 1)

    plt.xticks([])
    plt.ylabel("Similarity", fontsize=25)
    plt.xlabel("Time", fontsize=25)
    plt.tick_params(labelsize=20)
    ax.set_xticks((start_t, end_t-t_step))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()