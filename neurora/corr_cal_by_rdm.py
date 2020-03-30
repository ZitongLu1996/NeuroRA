# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Cosfficient between RDMs by different modes '

__author__ = 'Zitong Lu'

import numpy as np
import math
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.rdm_corr import rdm_correlation_kendall
from neurora.rdm_corr import rdm_similarity
from neurora.rdm_corr import rdm_distance

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the Similarity/Correlation Cosfficient between RDMs based on EEG/MEG/fNIRS/ECoG/electrophysiological data and a demo RDM'
def rdms_corr(demo_rdm, EEG_rdms, method="spearman", rescale=False):

    # the shape of EEG_rdms must be: [N_time_points, N_cons, N_cons] or [N_channels, N_cons, N_cons]

    N = np.shape(EEG_rdms)[0]

    corrs = np.zeros([N, 2], dtype=np.float64)

    for n in range(N):

        if method == "spearman":

            corrs[n] = rdm_correlation_spearman(demo_rdm, EEG_rdms[n], rescale=rescale)

        elif method == "pearson":

            corrs[n] = rdm_correlation_pearson(demo_rdm, EEG_rdms[n], rescale=rescale)

        elif method == "kendall":

            corrs[n] = rdm_correlation_kendall(demo_rdm, EEG_rdms[n], rescale=rescale)

        elif method == "similarity":

            corrs[n, 0] = rdm_similarity(demo_rdm, EEG_rdms[n], rescale=rescale)

        elif method == "distance":

            corrs[n, 0] = rdm_distance(demo_rdm, EEG_rdms[n], rescale=rescale)

    return corrs

' a function for calculating the Similarity/Correlation Cosfficient between RDMs based on fMRI data and a demo RDM'

def fmrirdms_corr(demo_rdm, fmri_rdms, method="spearman", rescale=False):


    # the shape of fmri_rdms must be: [n_x, n_y, n_z, N_cons, N_cons]

    cons = np.shape(demo_rdm)[0]

    n_x = np.shape(fmri_rdms)[0]
    n_y = np.shape(fmri_rdms)[1]
    n_z = np.shape(fmri_rdms)[2]

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

                    corrs[i, j, k] = rdm_correlation_spearman(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "pearson":

                    corrs[i, j, k] = rdm_correlation_pearson(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "kendall":

                    corrs[i, j, k] = rdm_correlation_kendall(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "similarity":

                    corrs[i, j, k, 0] = rdm_similarity(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

                elif method == "distance":

                    corrs[i, j, k, 0] = rdm_distance(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

                print(corrs[i, j, k])

    return np.abs(corrs)