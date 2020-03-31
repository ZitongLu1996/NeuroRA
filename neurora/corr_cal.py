# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Cosfficient between two different modes data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.rdm_cal import bhvRDM
from neurora.rdm_cal import eegRDM
from neurora.rdm_cal import ecogRDM
from neurora.rdm_cal import fmriRDM
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.rdm_corr import rdm_correlation_kendall
from neurora.rdm_corr import rdm_similarity
from neurora.rdm_corr import rdm_distance

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the Similarity/Correlation Coefficient between behavioral data and EEG/MEG/fNIRS data'
def bhvANDeeg_corr(bhv_data, EEG_data, sub_opt=0, bhv_data_opt=1, time_win=5, chl_opt=0, time_opt=0, method="spearman", rescale=False):

    subs = np.shape(bhv_data)[1]
    chls = np.shape(EEG_data)[3]
    ts = int(np.shape(EEG_data)[4]/time_win)

    if sub_opt == 1:

        if bhv_data_opt == 0:

            return None

        # if bhv_data_opt=1

        bhv_rdms = bhvRDM(bhv_data, sub_opt=sub_opt, data_opt=bhv_data_opt)

        if chl_opt == 0:

            if time_opt == 0:

                eeg_rdms = eegRDM(EEG_data, time_win=time_win, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

                corrs = np.zeros([subs, 2], dtype=np.float64)

                for i in range(subs):

                    if method == "spearman":

                        corrs[i] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                    elif method == "pearson":

                        corrs[i] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                    elif method == "kendall":

                        corrs[i] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                    elif method == "similarity":

                        corrs[i, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                    elif method == "distance":

                        corrs[i, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                return corrs

            # if time_opt=1

            eeg_rdms = eegRDM(EEG_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

            corrs = np.zeros([subs, ts, 2], dtype=np.float64)

            for i in range(subs):

                for j in range(ts):

                    if method == "spearman":

                        corrs[i, j] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                    elif method == "pearson":

                        corrs[i, j] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                    elif method == "kendall":

                        corrs[i, j] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                    elif method == "similarity":

                        corrs[i, j, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                    elif method == "distance":

                        corrs[i, j, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

            return corrs

        # chl_opt=1

        if time_opt == 1:
            return None

        # time_opt=0

        eeg_rdms = eegRDM(EEG_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

        corrs = np.zeros([subs, chls], dtype=np.float64)

        for i in range(subs):

            for j in range(chls):

                if method == "spearman":

                    corrs[i, j] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                elif method == "pearson":

                    corrs[i, j] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                elif method == "kendall":

                    corrs[i, j] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                elif method == "similarity":

                    corrs[i, j, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

                elif method == "distance":

                    corrs[i, j, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

        return corrs

    # if sub_opt=0

    bhv_rdm = bhvRDM(bhv_data, sub_opt=sub_opt, data_opt=bhv_data_opt)

    if chl_opt == 1:

        if time_opt == 1:

            eeg_rdms = eegRDM(EEG_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

            corrs = np.zeros([chls, ts, 2], dtype=np.float64)

            for i in range(chls):

                for j in range(ts):

                    if method == "spearman":

                        corrs[i, j] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

                    elif method == "pearson":

                        corrs[i, j] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

                    elif method == "kendall":

                        corrs[i, j] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

                    elif method == "similarity":

                        corrs[i, j, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

                    elif method == "distance":

                        corrs[i, j, 0] = rdm_distance(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

            return corrs

        # if time_opt=0

        eeg_rdms = eegRDM(EEG_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

        corrs = np.zeros([chls, 2], dtype=np.float64)

        for i in range(chls):

            if method == "spearman":

                corrs[i] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "pearson":

                corrs[i] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "kendall":

                corrs[i] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "similarity":

                corrs[i, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i])

            elif method == "distance":

                corrs[i, 0] = rdm_distance(bhv_rdm, eeg_rdms[i])

        return corrs

    # if chl_opt=0

    if time_opt == 1:

        eeg_rdms = eegRDM(EEG_data, sub_opt=0, chl_opt=0, time_opt=1)

        corrs = np.zeros([ts, 2], dtype=np.float64)

        for i in range(ts):

            if method == "spearman":

                corrs[i] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "pearson":

                corrs[i] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "kendall":

                corrs[i] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "similarity":

                corrs[i, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i], rescale=rescale)

            elif method == "distance":

                corrs[i, 0] = rdm_distance(bhv_rdm, eeg_rdms[i], rescale=rescale)

        return corrs

    # if time_opt=0

    eeg_rdm = eegRDM(EEG_data, sub_opt=0, chl_opt=0, time_opt=0)

    corr = np.zeros([2], dtype=np.float64)

    if method == "spearson":

        corr = rdm_correlation_spearman(bhv_rdm, eeg_rdm, rescale=rescale)

    elif method == "pearson":

        corr = rdm_correlation_pearson(bhv_rdm, eeg_rdm, rescale=rescale)

    elif method == "kendall":

        corr = rdm_correlation_kendall(bhv_rdm, eeg_rdm, rescale=rescale)

    elif method == "similarity":

        corr[0] = rdm_similarity(bhv_rdm, eeg_rdm, rescale=rescale)

    elif method == "distance":

        corr[0] = rdm_distance(bhv_rdm, eeg_rdm, rescale=rescale)

    return corr

' a function for calculating the Similarity/Correlation Cosfficient between behavioral data and sEEG/ECoG/eletricophysiological data'

def bhvANDecog_corr(bhv_data, ele_data, time_win=5, ecog_opt="allin", method="spearman", rescale=False):

    # sub_opt = 1, bhv_data here belongs to one subject, and its shape must be : [cons, trials]

    cons, trials = np.shape(bhv_data)
    ts = np.shape(ele_data)[3]

    bhv_data = np.reshape(bhv_data, [cons, 1, trials])

    bhv_rdm = np.reshape(bhvRDM(bhv_data, sub_opt=1, data_opt=1), [cons, cons])

    if ecog_opt == "channel":

        ecog_rdms = ecogRDM(ele_data, opt="channel")

        chls_num = np.shape(ele_data)[2]

        corrs = np.zeros([chls_num, 2], dtype=np.float64)

        for i in range(chls_num):

            if method == "spearman":

                corrs[i] = rdm_correlation_spearman(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "pearson":

                corrs[i] = rdm_correlation_pearson(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "kendall":

                corrs[i] = rdm_correlation_kendall(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "similarity":

                corrs[i, 0] = rdm_similarity(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "distance":

                corrs[i, 0] = rdm_distance(bhv_rdm, ecog_rdms[i], rescale=rescale)

        return corrs

    elif ecog_opt == "time":

        ecog_rdms = ecogRDM(ele_data, time_win=5, opt="time")

        ts = int(np.shape(ele_data)[3]/time_win)

        corrs = np.zeros([ts, 2], dtype=np.float64)

        for i in range(ts):

            if method == "spearman":

                corrs[i] = rdm_correlation_spearman(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "pearson":

                corrs[i] = rdm_correlation_pearson(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "kendall":

                corrs[i] = rdm_correlation_kendall(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "similarity":

                corrs[i, 0] = rdm_similarity(bhv_rdm, ecog_rdms[i], rescale=rescale)

            elif method == "distance":

                corrs[i, 0] = rdm_distance(bhv_rdm, ecog_rdms[i], rescale=rescale)

        return corrs

    # if ecog_opt="allin"

    ecog_rdm = ecogRDM(ele_data, opt="allin")

    corr = np.zeros([2], dtype=np.float64)

    if method == "spearman":

        corr = rdm_correlation_spearman(bhv_rdm, ecog_rdm, rescale=rescale)

    elif method == "pearson":

        corr = rdm_correlation_pearson(bhv_rdm, ecog_rdm, rescale=rescale)

    elif method == "kendall":

        corr = rdm_correlation_kendall(bhv_rdm, ecog_rdm, rescale=rescale)

    elif method == "similarity":

        corr[0] = rdm_similarity(bhv_rdm, ecog_rdm, rescale=rescale)

    elif method == "distance":

        corr[0] = rdm_distance(bhv_rdm, ecog_rdm, rescale=rescale)

    return corr

# -*- coding: utf-8 -*-

' a function for calculating the Similarity/Correlation Cosfficient between behavioral data and fMRI data'

def bhvANDfmri_corr(bhv_data, fmri_data, bhv_data_opt=1, ksize=[3, 3, 3], strides=[1, 1, 1], method="spearman", rescale=False):
    # sub_opt=1

    if bhv_data_opt == 0:

        bhv_rdm = bhvRDM(bhv_data, sub_opt=0, data_opt=0)

    # if bhv_data_opt=1

    else:

        bhv_rdm = bhvRDM(bhv_data, sub_opt=0, data_opt=1)

    print("****************")

    print("get behavior RDM")

    print(bhv_rdm)

    fmri_rdms = fmriRDM(fmri_data, ksize=ksize, strides=strides)

    print("****************")

    print("get fMRI RDM")

    print(np.shape(fmri_rdms))

    cons = np.shape(bhv_data)[0]

    nx = np.shape(fmri_data)[2]
    ny = np.shape(fmri_data)[3]
    nz = np.shape(fmri_data)[4]

    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                """index = 0

                for m in range(cons):

                    for n in range(cons):

                         if math.isnan(fmri_rdms[i, j, k, m, n]) == True:

                             index = index + 1

                if index != 0:

                    corrs[i, j, k, 0] = 0
                    corrs[i, j, k, 1] = 1"""

                if method == "spearman":

                    corrs[i, j, k] = rdm_correlation_spearman(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "pearson":

                    corrs[i, j, k] = rdm_correlation_pearson(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "kendall":

                    corrs[i, j, k] = rdm_correlation_kendall(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "similarity":

                    corrs[i, j, k, 0] = rdm_similarity(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "distance":

                    corrs[i, j, k, 0] = rdm_distance(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                print(corrs[i, j, k])

    return corrs

' a function for calculating the Similarity/Correlation Cosfficient between behavioral EEG/MEG/fNIRS and fMRI data'

def eegANDfmri_corr(eeg_data, fmri_data, chl_opt=0, ksize=[3, 3, 3], strides=[1, 1, 1], method="spearman", rescale=False):
    # sub_opt=0, time_opt=0

    nx = np.shape(fmri_data)[2]
    ny = np.shape(fmri_data)[3]
    nz = np.shape(fmri_data)[4]

    cons = np.shape(eeg_data)[0]

    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    sx = strides[0]
    sy = strides[1]
    sz = strides[0]

    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    fmri_rdms = fmriRDM(fmri_data, ksize=ksize, strides=strides)

    if chl_opt == 1:

        chls = np.shape(eeg_data)[3]

        eeg_rdms = eegRDM(eeg_data, sub_opt=0, chl_opt=1, time_opt=0)

        corrs = np.full([chls, n_x, n_y, n_z, 2], np.nan)

        for j in range(n_x):

            for k in range(n_y):

                for l in range(n_z):

                    """index = 0

                    for m in range(cons):

                        for n in range(cons):

                            if math.isnan(fmri_rdms[j, k, l]) == True:

                                index = index + 1"""

                    for i in range(chls):

                        """if index != 0:

                            corrs[i, j, k, l, 0] = 0
                            corrs[i, j, k, l, 1] = 1"""

                        if method == "spearman":

                            corrs[i, j, k, l] = rdm_correlation_spearman(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)

                        elif method == "pearson":

                            corrs[i, j, k, l] = rdm_correlation_pearson(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)

                        elif method == "kendall":

                            corrs[i, j, k, l] = rdm_correlation_kendall(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)

                        elif method == "similarity":

                            corrs[i, j, k, l, 0] = rdm_similarity(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)

                        elif method == "distance":

                            corrs[i, j, k, l, 0] = rdm_distance(eeg_rdms[i], fmri_rdms[i, j, k], rescale=rescale)

        return np.abs(corrs)

    # if chl_opt=0

    eeg_rdm = eegRDM(eeg_data, sub_opt=0, chl_opt=0, time_opt=0)

    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                """index = 0

                for m in range(cons):

                    for n in range(cons):

                        if math.isnan(fmri_rdms[i, j, k]) == True:

                            index = index + 1

                if index != 0:

                    corrs[i, j, k, 0] = 0
                    corrs[i, j, k, 1] = 1"""

                if method == "spearman":

                    corrs[i, j, k] = rdm_correlation_spearman(eeg_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "pearson":

                    corrs[i, j, k] = rdm_correlation_pearson(eeg_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "kendall":

                    corrs[i, j, k] = rdm_correlation_kendall(eeg_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "similarity":

                    corrs[i, j, k, 0] = rdm_similarity(eeg_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "distance":

                    corrs[i, j, k, 0] = rdm_distance(eeg_rdm, fmri_rdms[i, j, k], rescale=rescale)


    return np.abs(corrs)