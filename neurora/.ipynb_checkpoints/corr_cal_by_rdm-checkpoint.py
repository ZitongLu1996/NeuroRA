# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Cosfficient between RDMs by different modes '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.rdm_corr import rdm_correlation_spearman, rdm_correlation_pearson, rdm_correlation_kendall, \
    rdm_similarity, rdm_distance
from neurora.stuff import show_progressbar

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the similarity between RDMs based on RDMs of EEG-like data and a demo RDM'

def rdms_corr(demo_rdm, eeg_rdms, method="spearman", rescale=False, permutation=False, iter=1000):

    """
    Calculate the similarity between RDMs based on RDMs of EEG-like data and a demo RDM

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
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    Notes
    -----
    The demo RDM could be a behavioral RDM, a hypothesis-based coding model RDM or a computational model RDM.
    """

    if len(np.shape(demo_rdm)) != 2 or np.shape(demo_rdm)[0] != np.shape(demo_rdm)[1]:

        print("\nThe shape of the demo RDM should be [n_cons, n_cons].\n")

        return "Invalid input!"

    if len(np.shape(eeg_rdms)) < 2 or len(np.shape(eeg_rdms)) > 5 or np.shape(eeg_rdms)[-1] != np.shape(eeg_rdms)[-2]:

        print("\nThe shape of the EEG-like RDMs should be [n_cons, n_cons] or [n1, n_cons, n_cons] or "
              "[n1, n2, n_cons, n_cons] or [n1, n2, n3, n_cons, n_cons].\n")

        return "Invalid input!"

    if len(eeg_rdms.shape) == 5:

        print("\nComputing similarities")

        n1, n2, n3 = eeg_rdms.shape[:3]

        # initialize the corrs
        corrs = np.zeros([n1, n2, n3, 2])

        total = n1 * n2 * n3

        # calculate the corrs
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):

                    # show the progressbar
                    percent = (i * n2 * n2 + j * n3 + k + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    if method == "spearman":
                        corrs[i, j, k] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i, j, k], rescale=rescale,
                                                                  permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i, j, k] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i, j, k], rescale=rescale,
                                                                 permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i, j, k] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i, j, k], rescale=rescale,
                                                                 permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, j, k, 0] = rdm_similarity(demo_rdm, eeg_rdms[i, j, k], rescale=rescale)
                    elif method == "distance":
                        corrs[i, j, k, 0] = rdm_distance(demo_rdm, eeg_rdms[i, j, k], rescale=rescale)

        print("\nComputing finished!")

        return corrs

    if len(eeg_rdms.shape) == 4:

        print("\nComputing similarities")

        n1, n2 = eeg_rdms.shape[:2]

        # initialize the corrs
        corrs = np.zeros([n1, n2, 2])

        total = n1 * n2

        # calculate the corrs
        for i in range(n1):
            for j in range(n2):

                if method == "spearman":
                    corrs[i, j] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i, j], rescale=rescale,
                                                           permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i, j], rescale=rescale,
                                                          permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i, j], rescale=rescale,
                                                          permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, 0] = rdm_similarity(demo_rdm, eeg_rdms[i, j], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, 0] = rdm_distance(demo_rdm, eeg_rdms[i, j], rescale=rescale)

        print("\nComputing finished!")

        return corrs

    if len(eeg_rdms.shape) == 3:

        print("\nComputing similarities")

        n1 = eeg_rdms.shape[0]

        # initialize the corrs
        corrs = np.zeros([n1, 2])

        # calculate the corrs
        for i in range(n1):
            if method == "spearman":
                corrs[i] = rdm_correlation_spearman(demo_rdm, eeg_rdms[i], rescale=rescale, permutation=permutation,
                                                    iter=iter)
            elif method == "pearson":
                corrs[i] = rdm_correlation_pearson(demo_rdm, eeg_rdms[i], rescale=rescale, permutation=permutation,
                                                   iter=iter)
            elif method == "kendall":
                corrs[i] = rdm_correlation_kendall(demo_rdm, eeg_rdms[i], rescale=rescale, permutation=permutation,
                                                   iter=iter)
            elif method == "similarity":
                corrs[i, 0] = rdm_similarity(demo_rdm, eeg_rdms[i], rescale=rescale)
            elif method == "distance":
                corrs[i, 0] = rdm_distance(demo_rdm, eeg_rdms[i], rescale=rescale)

        print("\nComputing finished!")

        return corrs

    print("\nComputing the similarity")

    # initialize the corrs
    corr = np.zeros([2])

    # calculate the corrs
    if method == "spearman":
        corr = rdm_correlation_spearman(demo_rdm, eeg_rdms, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "pearson":
        corr = rdm_correlation_pearson(demo_rdm, eeg_rdms, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "kendall":
        corr = rdm_correlation_kendall(demo_rdm, eeg_rdms, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "similarity":
        corr[0] = rdm_similarity(demo_rdm, eeg_rdms, rescale=rescale)
    elif method == "distance":
        corr[0] = rdm_distance(demo_rdm, eeg_rdms, rescale=rescale)

    print("\nComputing finished!")

    return corr


' a function for calculating the similarity between fMRI searchlight RDMs and a demo RDM'

def fmrirdms_corr(demo_rdm, fmri_rdms, method="spearman", rescale=False, permutation=False, iter=1000):


    """
    Calculate the similarity between fMRI searchlight RDMs and a demo RDM

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
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between fMRI searchlight RDMs and a demo RDM
        The shape of RDMs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for searchlight
        along the x, y, z axis and 2 represents a r-value and a p-value.

    Notes
    -----
    The demo RDM could be a behavioral RDM, a hypothesis-based coding model RDM or a computational model RDM.
    """

    if len(np.shape(demo_rdm)) != 2 or np.shape(demo_rdm)[0] != np.shape(demo_rdm)[1]:

        print("\nThe shape of the demo RDM should be [n_cons, n_cons].\n")

        return "Invalid input!"

    if len(np.shape(fmri_rdms)) != 5 or np.shape(fmri_rdms)[3] != np.shape(fmri_rdms)[4]:

        print("\nThe shape of the fMRI RDMs should be [n_x, n_y, n_z, n_cons, n_cons].\n")

        return "Invalid input!"

    # calculate the number of the calculation units in the x, y, z directions
    n_x = np.shape(fmri_rdms)[0]
    n_y = np.shape(fmri_rdms)[1]
    n_z = np.shape(fmri_rdms)[2]

    print("\nComputing similarities")

    # initialize the corrs
    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    total = n_x * n_y * n_z

    # calculate the corrs
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                # show the progressbar
                percent = (i * n_y * n_z + j * n_z + k + 1) / total * 100
                show_progressbar("Calculating", percent)

                if method == "spearman":
                    corrs[i, j, k] = rdm_correlation_spearman(demo_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j, k] = rdm_correlation_pearson(demo_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j, k] = rdm_correlation_kendall(demo_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, k, 0] = rdm_similarity(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, k, 0] = rdm_distance(demo_rdm, fmri_rdms[i, j, k], rescale=rescale)

    print("\nComputing finished!")

    return corrs