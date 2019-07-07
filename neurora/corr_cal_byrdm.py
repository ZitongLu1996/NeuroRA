# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Cosfficient between RDMs by different modes '

__author__ = 'Zitong Lu'

import numpy as np
import math
from neurora.rsa_corr import rsa_correlation_spearman
from neurora.rsa_corr import rsa_correlation_pearson
from neurora.rsa_corr import rsa_correlation_kendall
from neurora.rsa_corr import rsa_similarity
from neurora.rsa_corr import rsa_distance

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the Similarity/Correlation Cosfficient between RDMs based on EEG/MEG/fNIRS data and a demo RDM'
def eegrdms_corr(demo_rdm, EEG_rdms, method="spearman"):

    # the shape of EEG_rdms must be: [N_time_points, N_cons, N_cons] or [N_channels, N_cons, N_cons]

    N = np.shape(EEG_rdms)[0]

    corrs = np.zeros([N, 2], dtype=np.float64)

    for n in range(N):

        if method == "spearman":

            corrs[n] = rsa_correlation_spearman(demo_rdm, EEG_rdms[n])

        elif method == "pearson":

            corrs[n] = rsa_correlation_pearson(demo_rdm, EEG_rdms[n])

        elif method == "kendall":

            corrs[n] = rsa_correlation_kendall(demo_rdm, EEG_rdms[n])

        elif method == "similarity":

            corrs[n, 0] = rsa_similarity(demo_rdm, EEG_rdms[n])

        elif method == "distance":

            corrs[n, 0] = rsa_distance(demo_rdm, EEG_rdms[n])

    return corrs

' a function for calculating the Similarity/Correlation Cosfficient between RDMs based on fMRI data and a demo RDM'

def fmrirdms_corr(demo_rdm, fmri_rdms, method="spearman"):


    # the shape of fmri_rdms must be: [n_x, n_y, n_z, N_cons, N_cons]

    cons = np.shape(demo_rdm)[0]

    n_x = np.shape(fmri_rdms)[0]
    n_y = np.shape(fmri_rdms)[1]
    n_z = np.shape(fmri_rdms)[2]

    corrs = np.zeros([n_x, n_y, n_z, 2], dtype=np.float64)

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                index = 0

                for m in range(cons):

                    for n in range(cons):

                         if math.isnan(fmri_rdms[i, j, k, m, n]) == True:

                             index = index + 1

                if index != 0:

                    corrs[i, j, k, 0] = 0
                    corrs[i, j, k, 1] = 1

                elif method == "spearman":

                    corrs[i, j, k] = rsa_correlation_spearman(bhv_rdm, fmri_rdms[i, j, k])

                elif method == "pearson":

                    corrs[i, j, k] = rsa_correlation_pearson(bhv_rdm, fmri_rdms[i, j, k])

                elif method == "kendall":

                    corrs[i, j, k] = rsa_correlation_kendall(bhv_rdm, fmri_rdms[i, j, k])

                elif method == "similarity":

                    corrs[i, j, k, 0] = rsa_similarity(bhv_rdm, fmri_rdms[i, j, k])

                elif method == "distance":

                    corrs[i, j, k, 0] = rsa_distance(bhv_rdm, fmri_rdms[i, j, k])

                print(corrs[i, j, k])

    return corrs