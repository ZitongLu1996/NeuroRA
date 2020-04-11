# -*- coding: utf-8 -*-

' a module for calculating the neural pattern similarity based neural data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr
import math

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the neural pattern similarity '

def nps(data, time_win=5, sub_opt=0):
    # shape of data: [2, n_subs, n_trials, n_chls, n_ts]
    #                 here 2 means calculating the nps between the neural activities under two conditions
    # time_win: the time-window, if time_win=5, that means each calculation process bases on 5 time points
    #           this is also a processing of downsampling
    # sub_opt : 0: consider all the subjects
    #           1: calculating independently for each subject

    nsubs, ntrials, nchls, nts = data.shape[1:]

    avgdata = np.average(data, axis=2)

    ts = int(nts/time_win)

    if sub_opt == 1:

        nps = np.zeros([nsubs, nchls, ts, 2])

        for sub in range(nsubs):

            for i in range(nchls):
                for j in range(ts):

                    data1 = avgdata[0, sub, i, j*time_win:j*time_win+time_win]
                    data2 = avgdata[1, sub, i, j*time_win:j*time_win+time_win]

                    nps[sub, i, j] = pearsonr(data1, data2)

        return nps

    # if sub_opt == 0

    nps = np.zeros([nchls, ts, 2])

    for i in range(nchls):
        for j in range(ts):

            data1 = avgdata[0, :, i, j*time_win:j*time_win+time_win]
            data2 = avgdata[1, :, i, j*time_win:j*time_win+time_win]

            data1 = np.reshape(data1, nsubs*time_win)
            data2 = np.reshape(data2, nsubs*time_win)

            nps[i, j] = pearsonr(data1, data2)

    return nps

def nps_fmri(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1]):
    # ksize=[kx, ky, kz] represents that the calculation unit contains k1*k2*k3 voxels
    # strides=[sx, sy, sz] represents the moving steps along the x, y, z
    # the shape of fmri_data : [2, N_subs, nx, ny, nz]
    # here 2 means calculating the nps between the neural activities under two conditions
    # N_subs, nx, ny, nz represent the number of subjects, the size of the fmri data

    nsubs, nx, ny, nz = np.shape(fmri_data)[1:]

    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    sx = ksize[0]
    sy = ksize[1]
    sz = ksize[2]

    # calculate the number of the calculation units
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    data = np.full([n_x, n_y, n_z, 2, kx*ky*kz, nsubs], np.nan)

    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                for i in range(2):
                    index = 0
                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):
                                for j in range(nsubs):

                                    data[x, y, z, i, index, j] = fmri_data[i, j, x+k1, y+k2, z+k3]

                                index = index + 1

    data = np.reshape(data, [n_x, n_y, n_z, 2, kx*ky*kz*nsubs])

    nps = np.full([n_x, n_y, n_z, 2], np.nan)

    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                nps[x, y, z] = np.abs(pearsonr(data[x, y, z, 0], data[x, y, z, 1]))

    return nps

def nps_fmri_roi(fmri_data, mask_data):
    # the shape of fmri_data : [2, N_subs, nx, ny, nz]
    # here 2 means calculating the nps between the neural activities under two conditions
    # the shape of mask_data : [nx, ny, nz]

    nsubs, nx, ny, nz = fmri_data.shape[1:]

    n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                    n = n + 1

    data = np.zeros([2, nsubs, n], dtype=np.float)
    for p in range(2):
        for q in range(nsubs):
            n = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                            data[p, q, n] = fmri_data[p, q, i, j, k]
                            n = n + 1

    data = np.reshape(data, [2, nsubs*n])
    nps = np.abs(pearsonr(data[0], data[1]))

    return nps