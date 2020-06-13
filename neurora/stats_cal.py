# -*- coding: utf-8 -*-

' a module for calculating the Statistical Results '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from neurora.stuff import permutation_test


' a function for calculating the statistical results '

def stats(corrs, permutation=True, iter=5000):

    """
    Calculate the statistical results

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_subs, n_chls, n_ts, 2]. n_subs, n_chls, n_ts represent the number of subjects, the
        number of channels and the number of time-points. 2 represents a r-value and a p-value.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    stats : array
        The statistical results.
        The shape of stats is [n_chls, n_ts, 2]. n_chls, n_ts represent the number of channels and the number of
        time-points. 2 represents a t-value and a p-value.

    Notes
    -----
    n_subs must >= 6.
    This function can be used for the correlation results of NPS, ISC, eeg-like RDMs-correlations.
    """

    # get the number of subjects, channels & time-points
    subs, chls, ts = np.shape(corrs)[:3]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # initialize the corrs
    stats = np.zeros([chls, ts, 2], dtype=np.float)

    # get r-map
    rs = corrs[:, :, :, 0]

    #print(zs)

    # calculate the statistical results
    for i in range(chls):
        for j in range(ts):
            # t test
            stats[i, j] = ttest_1samp(rs[:, i, j], 0)

            if permutation == True:

                # Fisher r to z
                zs = 0.5*np.log((1+rs)/(1-rs))

                stats[i, j, 1] = permutation_test(zs[:, i, j], np.zeros([subs]), iter=iter)

    return stats


' a function for calculating the statistical results for fMRI '

def stats_fmri(corrs, permutation=True, iter=5000):

    """
    Calculate the statistical results for fMRI

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    stats : array
        The statistical results.
        The shape of stats is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for
        searchlight along the x, y, z axis and 2 represents a t-value and a p-value.

    Notes
    -----
    n_subs must >= 6.
    This function can be used for the results of searchlight fMRI NPS and searchlight fMRI RDM-correlations.
    """

    # get the number of subjects
    subs = np.shape(corrs)[0]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs)[1:4]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2], dtype=np.float)

    # get r-map
    rs = corrs[:, :, :, :, 0]

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                # t test
                stats[i, j, k] = ttest_1samp(rs[:, i, j, k], 0)

                if permutation == True:
                    # Fisher r to z
                    zs = 0.5 * np.log((1 + rs) / (1 - rs))
                    stats[i, j, k, 1] = permutation_test(zs[:, i, j, k], np.zeros([subs]), iter=iter)

    return stats


' a function for calculating the statistical results for fMRI (ISC searchlight) '

def stats_iscfmri(corrs, permutation=True, iter=5000):

    """
    Calculate the statistical results for fMRI (ISC searchlight)

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_ts, n_subs!/(2!*(n_subs-2)!), n_x, n_y, n_z, 2]. n_ts, n_subs, n_x, n_y, n_z
        represent the number of subjects, the number of calculation units for searchlight along the x, y, z axis and 2
        represents a r-value and a p-value.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    stats : array
        The statistical results.
        The shape of stats is [n_ts, n_x, n_y, n_z, 2]. n_ts, n_x, n_y, n_z represent the number of time-points, the
        number of calculation units for searchlight along the x, y, z axis and 2 represents a t-value and a p-value.

    Notes
    -----
    n_subs must >= 4 (n_subs!/(2!*(n_subs-2)!) >= 6).
    """

    # get the number of time-points, pairs
    ts, npairs = np.shape(corrs)[:2]

    # n_subs!/(2!*(n_subs-2)!)>=6
    if npairs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs)[2:5]

    # initialize the corrs
    stats = np.zeros([ts, n_x, n_y, n_z, 2], dtype=np.float)

    # get r-map
    rs = corrs[:, :, :, :, :, 0]

    # calculate the statistical results
    for t in range(ts):
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    # t test
                    stats[t, i, j, k] = ttest_1samp(rs[t, :, i, j, k], 0)

                    if permutation == True:

                        # Fisher r to z
                        zs = 0.5 * np.log((1 + rs) / (1 - rs))
                        stats[t, i, j, k, 1] = permutation_test(zs[t, :, i, j, k], np.zeros([npairs]), iter=iter)

    return stats


' a function for calculating the statistical results (for STPS) '

def stats_stps(corrs1, corrs2, permutation=True, iter=5000):

    """
    Calculate the statistical results （for STPS）

    Parameters
    ----------
    corrs1 : array
        The correlation coefficients under condition1.
        The shape of corrs1 must be [n_subs, n_chls, n_ts]. n_subs, n_chls, n_ts represent the number of subjects, the
        number of channels and the number of time-points.
    corrs2 : array
        The correlation coefficients under condition2.
        The shape of corrs2 must be [n_subs, n_chls, n_ts]. n_subs, n_chls, n_ts represent the number of subjects, the
        number of channels and the number of time-points.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    stats : array
        The statistical results.
        The shape of stats is [n_chls, n_ts, 2]. n_chls, n_ts represent the number of channels and the number of
        time-points. 2 represents a t-value and a p-value.

    Notes
    -----
    n_subs must >= 6.
    """

    # get the number of subjects, channels & time-points
    subs, chls, ts = np.shape(corrs1)

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # initialize the corrs
    stats = np.zeros([chls, ts, 2], dtype=np.float)

    # get r-map
    rs1 = corrs1
    rs2 = corrs2

    # calculate the statistical results
    for i in range(chls):
        for j in range(ts):
            # t test
            stats[i, j] = ttest_rel(rs1[:, i, j], rs2[:, i, j])

            if permutation == True:

                # Fisher r to z
                zs1 = 0.5*np.log((1+rs1)/(1-rs1))
                zs2 = 0.5*np.log((1+rs2)/(1-rs2))

                stats[i, j, 1] = permutation_test(zs1[:, i, j], zs2[:, i, j], iter=iter)

    return stats


' a function for calculating the statistical results for fMRI (STPS searchlight) '

def stats_stpsfmri(corrs1, corrs2, permutation=True, iter=5000):

    """
    Calculate the statistical results for fMRI (STPS searchlight)

    Parameters
    ----------
    corrs1 : array
        The correlation coefficients under condition1.
        The shape of corrs1 must be [n_subs, n_x, n_y, n_z]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis.
    corrs2 : array
        The correlation coefficients under condition2.
        The shape of corrs2 must be [n_subs, n_x, n_y, n_z]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    stats : array
        The statistical results.
        The shape of stats is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for
        searchlight along the x, y, z axis and 2 represents a t-value and a p-value.

    Notes
    -----
    n_subs must >= 6.
    """

    # get the number of subjects
    subs = np.shape(corrs1)[0]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs1)[1:]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2], dtype=np.float)

    # get r-map
    rs1 = corrs1
    rs2 = corrs2

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                # t test
                stats[i, j, k] = ttest_rel(rs1[:, i, j, k], rs2[:, i, j, k])

                if permutation == True:

                    # Fisher r to z
                    zs1 = 0.5 * np.log((1 + rs1) / (1 - rs1))
                    zs2 = 0.5 * np.log((1 + rs2) / (1 - rs2))
                    stats[i, j, k, 1] = permutation_test(zs1[:, i, j, k], zs2[:, i, j, k], iter=iter)

    return stats



