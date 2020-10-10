# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Cosfficient between RDMs by different modes '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.rdm_corr import rdm_correlation_kendall
from neurora.rdm_corr import rdm_similarity
from neurora.rdm_corr import rdm_distance

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the Similarity/Correlation Cosfficient between RDMs based on EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDMs and a demo RDM'

def rdms_corr(demo_rdm, eeg_rdms, method="spearman", fisherz=False, rescale=False, permutation=False, iter=5000):

    """
    Calculate the Similarities between EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDMs and a demo RDM

    Parameters
    ----------
    demo_rdm : array [n_cons, n_cons]
        A demo RDM.
    eeg_rdms : array
        The EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDM(s).
        The shape can be [n_cons, n_cons] or [n1, n_cons, n_cons] or [n1, n2, n_cons, n_cons] or
        [n1, n2, n3, n_cons, n_cons]. ni(i=1, 2, 3) can be int(n_ts/timw_win), n_chls, n_subs.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the RDMs or not.
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDMs and a demo RDM
        If the shape of eeg_rdms is [n_cons, n_cons], the shape of corrs will be [2]. If the shape of eeg_rdms is
        [n1, n_cons, n_cons], the shape of corrs will be [n1, 2]. If the shape of eeg_rdms is [n1, n2, n_cons, n_cons],
        the shape of corrs will be [n1, n2, 2]. If the shape of eeg_rdms is [n1, n2, n3, n_cons, n_cons], the shape of
        corrs will be [n1, n2, n3, 2]. ni(i=1, 2, 3) can be int(n_ts/timw_win), n_chls, n_subs. 2 represents a r-value
        and a p-value.
    """

    if len(eeg_rdms.shape) == 5:

        n1, n2, n3 = eeg_rdms.shape[:3]

        # initialize the corrs
        corrs = np.zeros([n1, n2, n3, 2], dtype=np.float64)

        # calculate the corrs
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):

                    if method == "spearman":
                        corrs[i, j, k] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i, j, k] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i, j, k] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, j, k, 0] = rdm_similarity(demo_rdm, eeg_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "distance":
                        corrs[i, j, k, 0] = rdm_distance(demo_rdm, eeg_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)

        return corrs

    if len(eeg_rdms.shape) == 4:

        n1, n2 = eeg_rdms.shape[:2]

        # initialize the corrs
        corrs = np.zeros([n1, n2, 2], dtype=np.float64)

        # calculate the corrs
        for i in range(n1):
            for j in range(n2):

                if method == "spearman":
                    corrs[i, j] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, 0] = rdm_similarity(demo_rdm, eeg_rdms[i, j], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, 0] = rdm_distance(demo_rdm, eeg_rdms[i, j], rescale=rescale)

        return corrs

    if len(eeg_rdms.shape) == 3:

        n1 = eeg_rdms.shape[0]

        # initialize the corrs
        corrs = np.zeros([n1, 2], dtype=np.float64)

        # calculate the corrs
        for i in range(n1):
            if method == "spearman":
                corrs[i] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "pearson":
                corrs[i] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "kendall":
                corrs[i] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "similarity":
                corrs[i, 0] = rdm_similarity(demo_rdm, eeg_rdms[i], rescale=rescale)
            elif method == "distance":
                corrs[i, 0] = rdm_distance(demo_rdm, eeg_rdms[i], rescale=rescale)

        return corrs

    # initialize the corrs
    corr = np.zeros([2], dtype=np.float64)

    # calculate the corrs
    if method == "spearman":
        corr = rdm_correlation_spearman(demo_rdm, eeg_rdms, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "pearson":
        corr = rdm_correlation_pearson(demo_rdm, eeg_rdms, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "kendall":
        corr = rdm_correlation_kendall(demo_rdm, eeg_rdms, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "similarity":
        corr[0] = rdm_similarity(demo_rdm, eeg_rdms, rescale=rescale)
    elif method == "distance":
        corr[0] = rdm_distance(demo_rdm, eeg_rdms, rescale=rescale)

    return corr


' a function for calculating the Similarity/Correlation Cosfficient between fMRI RDMs and a demo RDM'

def fmrirdms_corr(demo_rdm, fmri_rdms, method="spearman", fisherz=False, rescale=False, permutation=False, iter=5000):


    """
    Calculate the Similarities between fMRI searchlight RDMs and a demo RDM

    Parameters
    ----------
    demo_rdm : array [n_cons, n_cons]
        A demo RDM.
    fmri_rdms : array
        The fMRI-Searchlight RDMs.
        The shape of RDMs is [n_x, n_y, n_z, n_cons, n_cons]. n_x, n_y, n_z represent the number of calculation units
        for searchlight along the x, y, z axis.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the RDMs or not.
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between fMRI searchlight RDMs and a demo RDM
        The shape of RDMs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for searchlight
        along the x, y, z axis and 2 represents a r-value and a p-value.
    """
    # calculate the number of the calculation units in the x, y, z directions
    n_x = np.shape(fmri_rdms)[0]
    n_y = np.shape(fmri_rdms)[1]
    n_z = np.shape(fmri_rdms)[2]

    # initialize the corrs
    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    # calculate the corrs
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                if method == "spearman":
                    corrs[i, j, k] = rdm_correlation_spearman(demo_rdm, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j, k] = rdm_correlation_pearson(demo_rdm, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j, k] = rdm_correlation_kendall(demo_rdm, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, k, 0] = rdm_similarity(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, k, 0] = rdm_distance(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

    return corrs