# -*- coding: utf-8 -*-

' a module for calculating the RDM based on multimode neural data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.stuff import limtozero

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the RDM based on behavioral data '
def bhvRDM(bhv_data, sub_opt=0, data_opt=1):
    # sub_opt : 0 : return only one rdm
    #           1 : each subject obtains a rdm
    # data_opt : 1 : for raw data, each subject's each trial has a value of data
    #            0 : each subject has had a value of data, ignore the effect of trials

    cons = len(bhv_data)  # get the number of conditions

    n_subs = []  # save the number of subjects of each condition

    for i in range(cons):
        n_subs.append(np.shape(bhv_data[i])[0])

    subs = n_subs[0]

    if data_opt == 0:   # shape of bhv_data: [N_cons, N_subs]
                        # N_cons & N_subs represent the number of conditions & the number of subjects

        if sub_opt == 1:
            return None

        if len(set(n_subs)) != 1:
            return None

        rdm = np.zeros([cons, cons], dtype=np.float64)

        for i in range(cons):
            for j in range(cons):
                r = np.corrcoef(bhv_data[i], bhv_data[j])[0][1]  # calculate the Pearson Coefficient
                rdm[i, j] = limtozero(1 - abs(r)) # calculate the dissimilarity

        return rdm

    # if data_opt=1
                      # shape of bhv_data: [N_cons, N_subs, N_trials]
                      # N_cons, N_subs, N_trials represent the number of conditions, the number of subjects,
                      # the number of trials, respectively
    n_trials = []  # save the number of trials of each condition

    for i in range(cons):
        n_trials.append(np.shape(bhv_data[i])[1])

    if len(set(n_trials)) != 1:  # judge whether numbers of trials of different conditions are same
            return None

    if sub_opt == 1:

        rdms = np.zeros([subs, cons, cons], dtype=np.float64)

        for sub in subs:
            for i in range(cons):
                for j in range(cons):
                    r = np.corrcoef(bhv_data[i][sub], bhv_data[j][sub])[0][1]  # calculate the Pearson Coefficient
                    rdms[sub, i, j] = limtozero(1 - abs(r))  # calculate the dissimilarity

        return rdms


    # if sub_opt=0

    rdm = np.zeros([cons, cons], dtype=np.float64)  # initialize a con*con matrix as the rdm

    if len(set(n_subs)) != 1:  # judge whether numbers of trials of different conditions are same
        return None

    data = np.zeros([cons, subs], dtype=np.float64)

    for i in range(cons):
        for j in range(subs):
            data[i][j] = np.average(bhv_data[i][j])  # save the data for each subject under each condition, average the trials

    for i in range(cons):
        for j in range(cons):
            r = np.corrcoef(data[i], data[j])[0][1]  # calculate the Pearson Coefficient
            rdm[i, j] = limtozero(1 - abs(r))  # calculate the dissimilarity

    return rdm

' a function for calculating the RDM based on EEG/MEG/fNIRS data '

def eegRDM(EEG_data, sub_opt=0, chl_opt=0, time_opt=0):
    # shape of EEG_data: [n_cons, n_subs,_n_trials, n_chls, n_ts]
    # n_cons, n_subs, n_trials, n_chls, n_frqs, n_ts represent the number of
    # the conditions, subjects, trials, channels, frequencies and time-points
    # sub_opt : 0 : return only one rdm
    #           1 : each subject obtains a rdm
    # chl_opt : 0 : consider all the channels
    #           1 : each channel obtains a rdm
    # time_opt : 0 : consider the time sequence
    #            1 : each time-point obtains a rdm

    cons, subs, trials, chls, ts = np.shape(EEG_data)  # get the number of conditions, subjects, trials,
                                                       # channels and time points

    n_subs = []
    n_trials = []
    n_chls = []
    n_ts = []

    for i in range(cons):
        n_subs.append(np.shape(EEG_data[i])[0])
        n_trials.append(np.shape(EEG_data[i])[1])
        n_chls.append(np.shape(EEG_data[i])[2])
        n_ts.append(np.shape(EEG_data[i])[3])

    if len(set(n_chls)) != 1:
        return None

    if len(set(n_ts)) != 1:
        return None

    if sub_opt == 1:

        if time_opt == 1:

            if chl_opt == 1:
                return None

            data = np.zeros([subs, ts, cons, chls], dtype=np.float64)

            for i in range(subs):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(chls):
                            data[i, j, k, l] = np.average(EEG_data[k, i, :, l, j])

            rdms = np.zeros([subs, ts, cons, cons], dtype=np.float64)

            for i in range(subs):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(cons):
                            r = np.corrcoef(data[i, j, k], data[i, j, l])[0][1]
                            rdms[i, j, k, l] = limtozero(1 - abs(r))

            return rdms

        # if time_opt = 0

        if chl_opt == 1:

            data = np.zeros([subs, chls, cons, ts], dtype=np.float64)

            for i in range(subs):
                for j in range(chls):
                    for k in range(cons):
                        for l in range(ts):
                            data[i, j, k, l] = np.average(EEG_data[k, i, :, j, l])

            rdms = np.zeros([subs, chls, cons, cons], dtype=np.float64)

            for i in range(subs):
                for j in range(chls):
                    for k in range(cons):
                        for l in range(cons):
                            r = np.corrcoef(data[i, j, k], data[i, j, l])[0][1]
                            rdms[i, j, k, l] = limtozero(1 - abs(r))

            return rdms

        # if chl_opt = 0

        data = np.zeros([subs, cons, chls, ts], dtype=np.float64)

        for i in range(subs):
            for j in range(cons):
                for k in range(chls):
                    for l in range(ts):
                        data[i, j, k, l] = np.average(EEG_data[j, i, :, k, j])

        data = np.reshape(data, [subs, cons, chls*ts])

        rdms = np.zeros([subs, cons, cons], dtype=np.float64)

        for i in range(subs):

            for j in range(cons):

                for k in range(cons):

                    r = np.corrcoef(data[i, j], data[i, k])[0][1]

                    rdms[i, j, k] = limtozero(1 - abs(r))

        return rdms

    # if sub_opt = 0

    if len(set(n_subs)) != 1:
        return None

    if time_opt == 1:

        if chl_opt == 1:
            return None

        data = np.zeros([ts, cons, subs, chls], dtype=np.float64)

        for i in range(ts):

            for j in range(cons):

                for k in range(subs):

                    for l in range(chls):

                        data[i, j, k, l] = np.average(EEG_data[j, k, :, l, i])

        data = np.reshape(data, [ts, cons, subs * chls])

        rdms = np.zeros([ts, cons, cons], dtype=np.float64)

        for i in range(ts):

            for j in range(cons):

                for k in range(cons):

                    r = np.corrcoef(data[i, j], data[i, k])[0][1]

                    rdms[i, j, k] = limtozero(1 - abs(r))

        return rdms

    # if time_opt = 0

    if chl_opt == 1:

        data = np.zeros([chls, cons, subs, ts], dtype=np.float64)

        for i in range(chls):

            for j in range(cons):

                for k in range(subs):

                    for l in range(ts):

                        data[i, j, k, l] = np.average(EEG_data[j, k, :, i, l])

        data = np.reshape(data, [chls, cons, subs*ts])

        rdms = np.zeros([chls, cons, cons], dtype=np.float64)

        for i in range(chls):

            for j in range(cons):

                for k in range(cons):

                    r = np.corrcoef(data[i, j], data[i, k])[0][1]

                    rdms[i, j, k] = limtozero(1 - abs(r))

        return rdms

    # if chl_opt = 0

    data = np.zeros([cons, subs, chls, ts], dtype=np.float64)

    for i in range(cons):

        for j in range(subs):

            for k in range(chls):

                for l in range(ts):

                    data[i, j, k, l] = np.average(EEG_data[i, j, :, k, j])

    data = np.reshape(data, [cons, subs * chls * ts])

    rdm = np.zeros([cons, cons], dtype=np.float64)

    for i in range(cons):

        for j in range(cons):

            r = np.corrcoef(data[i], data[j])[0][1]

            rdm[i, j] = limtozero(1 - abs(r))

    return rdm

' a function for calculating the RDM based on ECoG/electrophysiological data '

def ecogRDM(ele_data, chls_num, opt="all"):
    # all these calculations belong to only one subject
    # chls_num represent the number of channels of the data
    # the shape of ele_data : [N_cons, N_trials, N_chls, N_ts]
    # N_cons, N_trials, N_chls, N_ts represent the number of conditions,
    # the number of trials, the number of channels, the number of time-points

    cons, trials, chls, ts = np.shape(ele_data)  #  get the number of conditins, trials, channels and time points

    n_trials = []
    n_chls = []
    n_ts = []

    for i in range(cons):

        n_trials.append(np.shape(ele_data[i])[0])
        n_chls.append(np.shape(ele_data[i])[1])
        n_ts.append(np.shape(ele_data[i])[2])

    if len(set(n_chls)) != 1:
        return None

    if len(set(n_ts)) != 1:
        return None

    if n_chls[0] != chls_num:
        return None

    if opt == "channel":

        data = np.zeros([chls, cons, ts], dtype=np.float64)

        for i in range(chls):

            for j in range(cons):

                for k in range(ts):

                    data[i, j, k] = np.average(ele_data[j, :, i, k])

        rdms = np.zeros([chls, cons, cons], dtype=np.float64)

        for i in range(chls):

            for j in range(cons):

                for k in range(cons):

                    r = np.corrcoef(data[i, j], data[i, k])[0][1]

                    rdms[i, j, k] = limtozero(1 - abs(r))

        return rdms

    elif opt == "time":

        data = np.zeros([ts, cons, chls], dtype=np.float64)

        for i in range(ts):

            for j in range(cons):

                for k in range(chls):

                    data[i, j, k] = np.average(ele_data[j, :, k, i])

        rdms = np.zeros([ts, cons, cons], dtype=np.float64)

        for i in range(ts):

            for j in range(cons):

                for k in range(cons):

                    r = np.corrcoef(data[i, j], data[i, k])[0][1]

                    rdms[i, j, k] = limtozero(1 - abs(r))

        return rdms

    # if opt = "allin"

    data = np.zeros([cons, chls, ts], dtype=np.float64)

    for i in range(cons):

        for j in range(chls):

            for k in range(ts):

                data[i, j, k] = np.average(ele_data[i, :, j, k])

    data = np.reshape(data, [cons, chls*ts])

    rdm = np.zeros([cons, cons], dtype=np.float64)

    for i in range(cons):

        for j in range(cons):

            r = np.corrcoef(data[i], data[j])[0][1]

            rdm[i, j] = limtozero(1 - abs(r))

    return rdm

' a function for calculating the RDM based on fMRI data '

def fmriRDM(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1]):
    # ksize=[kx, ky, kz] represents that the calculation unit contains k1*k2*k3 voxels
    # strides=[sx, sy, sz] represents the moving steps along the x, y, z
    # the shape of fmri_rdm : [N_cons, N_subs, nx, ny, nz]
    # N_cons, N_subs, nx, ny, nz represent the number of conditions,
    # the number of subjects, the size of the fmri data

    cons, subs, nx, ny, nz = np.shape(fmri_data)  # get the number of conditions, subjects and the size of the data

    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    n_subs = []
    n_trials = []

    for i in range(cons):

        n_subs.append(fmri_data[i][0])
        n_trials.append(fmri_data[i][1])

    # calculate the number of the calculation units
    n_x = int((nx - kx) / sx)+1
    n_y = int((ny - ky) / sy)+1
    n_z = int((nz - kz) / sz)+1

    data = np.zeros([n_x, n_y, n_z, cons, kx*ky*kz, subs], dtype=np.float64)

    for x in range(n_x):

        for y in range(n_y):

            for z in range(n_z):

                for i in range(cons):

                    for k1 in range(kx):

                        for k2 in range(ky):

                            for k3 in range(kz):

                                for j in range(subs):

                                    data[x, y, z, i, k1*kx+k2*ky+k3*ky, j] = fmri_data[i, j, x+k1, y+k2, z+k3]

    data = np.reshape(data, [n_x, n_y, n_z, cons, kx*ky*kz*subs])

    rdms = np.zeros([n_x, n_y, n_z, cons, cons], dtype=np.float64)

    for x in range(n_x):

        for y in range(n_y):

            for z in range(n_z):

                for i in range(cons):

                    for j in range(cons):

                        r = np.corrcoef(data[x, y, z, i], data[x, y, z, j])[0][1]

                        rdms[x, y, z, i, j] = limtozero(1 - abs(r))

                        print(rdms[x, y, z, i, j])

    return rdms

