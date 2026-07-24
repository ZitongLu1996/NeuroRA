# -*- coding: utf-8 -*-

' a module for conducting the statistical analysis '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import ttest_1samp, ttest_rel, ttest_ind
from neurora.stuff import permutation_test


' a function for conducting the statistical analysis for results of EEG-like data '

def stats(corrs, fisherz=True, permutation=True, iter=1000):

    """
    Conduct the statistical analysis for results of EEG-like data

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_subs, n_chls, n_ts, 2]. n_subs, n_chls, n_ts represent the number of subjects, the
        number of channels and the number of time-points. 2 represents a r-value and a p-value.
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs)) != 4:

        return "Invalid input!"

    # get the number of subjects, channels & time-points
    subs, chls, ts = np.shape(corrs)[:3]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # initialize the corrs
    stats = np.zeros([chls, ts, 2])

    # get r-map
    rs = corrs[:, :, :, 0]

    if fisherz == True:
        rs = 0.5 * np.log((1 + rs) / (1 - rs))
    #print(zs)

    # calculate the statistical results
    for i in range(chls):
        for j in range(ts):

            # t test
            stats[i, j] = ttest_1samp(rs[:, i, j], 0, alternative="greater")

            if permutation == True:

                stats[i, j, 1] = permutation_test(rs[:, i, j], np.zeros([subs]), iter=iter)

    return stats


' a function for conducting the statistical analysis for results of fMRI data (searchlight) '

def stats_fmri(corrs, fisherz=True, permutation=False, iter=1000):

    """
    Conduct the statistical analysis for results of fMRI data (searchlight)

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs)) != 5:

        return "Invalid input!"

    # get the number of subjects
    subs = np.shape(corrs)[0]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs)[1:4]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2])

    # get r-map
    rs = corrs[:, :, :, :, 0]

    if fisherz is True:

        rs = 0.5 * np.log((1+rs)/(1-rs))

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                # t test
                stats[i, j, k] = ttest_1samp(rs[:, i, j, k], 0, alternative="greater")

                if permutation == True:

                    stats[i, j, k, 1] = permutation_test(rs[:, i, j, k], np.zeros([subs]), iter=iter)

    return stats


' a function for conducting the statistical analysis for results of fMRI data (searchlight) within group '

def stats_fmri_compare_withingroup(corrs1, corrs2, fisherz=True, permutation=False, iter=1000):

    """
    Conduct the statistical analysis for results of fMRI data (searchlight) (within group: corrs1 > corrs2)

    Parameters
    ----------
    corrs1 : array
        The correlation coefficients under condition 1.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    corrs2 : array
        The correlation coefficients under condition 2.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs1)) != 5 or len(np.shape(corrs2)) != 5:

        return "Invalid input!"

    # get the number of subjects
    subs = np.shape(corrs1)[0]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs1)[1:4]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2])

    # get r-map
    rs1 = corrs1[:, :, :, :, 0]
    rs2 = corrs2[:, :, :, :, 0]

    if fisherz is True:

        rs1 = 0.5 * np.log((1+rs1)/(1-rs1))
        rs2 = 0.5 * np.log((1+rs2)/(1-rs2))

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                # t test
                stats[i, j, k] = ttest_rel(rs1[:, i, j, k], rs2[:, i, j, k], alternative="greater")

                if permutation == True:

                    stats[i, j, k, 1] = permutation_test(rs1[:, i, j, k], rs2[:, i, j, k], iter=iter)

    return stats


' a function for conducting the statistical analysis for results of fMRI data (searchlight) between two groups'

def stats_fmri_compare_betweengroups(corrs1, corrs2, fisherz=True, permutation=False, iter=5000):

    """
    Conduct the statistical analysis for results of fMRI data (searchlight) (between 2 groups: group1 > group2)

    Parameters
    ----------
    corrs1 : array
        The correlation coefficients for group 1.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    corrs2 : array
        The correlation coefficients for group 2.
        The shape of corrs must be [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent the number of subjects,
        the number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
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

    if len(np.shape(corrs1)) != 5 or len(np.shape(corrs2)) != 5:
        return "Invalid input!"

    # get the number of subjects
    subs1 = np.shape(corrs1)[0]
    subs2 = np.shape(corrs2)[0]

    # subs>=6
    if subs1 < 6 or subs2 < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs1)[1:4]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2])

    # get r-map
    rs1 = corrs1[:, :, :, :, 0]
    rs2 = corrs2[:, :, :, :, 0]

    if fisherz == True:
        rs1 = 0.5 * np.log((1 + rs1) / (1 - rs1))
        rs2 = 0.5 * np.log((1 + rs2) / (1 - rs2))

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                # t test
                stats[i, j, k] = ttest_ind(rs1[:, i, j, k], rs2[:, i, j, k], alternative="greater")

                if permutation == True:
                    stats[i, j, k, 1] = permutation_test(rs1[:, i, j, k], rs2[:, i, j, k], iter = iter)

    return stats


' a function for conducting the statistical analysis for results of fMRI data (ISC searchlight) '

def stats_iscfmri(corrs, fisherz=True, permutation=False, iter=1000):

    """
    Conduct the statistical analysis for results of fMRI data (ISC searchlight)

    Parameters
    ----------
    corrs : array
        The correlation coefficients.
        The shape of corrs must be [n_ts, n_subs!/(2!*(n_subs-2)!), n_x, n_y, n_z, 2]. n_ts, n_subs, n_x, n_y, n_z
        represent the number of subjects, the number of calculation units for searchlight along the x, y, z axis and 2
        represents a r-value and a p-value.
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs)) != 6:

        return "Invalid input!"

    # get the number of time-points, pairs
    ts, npairs = np.shape(corrs)[:2]

    # n_subs!/(2!*(n_subs-2)!)>=6
    if npairs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs)[2:5]

    # initialize the corrs
    stats = np.zeros([ts, n_x, n_y, n_z, 2])

    # get r-map
    rs = corrs[:, :, :, :, :, 0]

    if fisherz == True:
        # Fisher r to z
        rs = 0.5 * np.log((1 + rs) / (1 - rs))

    # calculate the statistical results
    for t in range(ts):
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):

                    # t test
                    stats[t, i, j, k] = ttest_1samp(rs[t, :, i, j, k], 0, alternative="greater")

                    if permutation == True:

                        stats[t, i, j, k, 1] = permutation_test(rs[t, :, i, j, k], np.zeros([npairs]), iter=iter)

    return stats


' a function for conducting the statistical analysis for results of EEG-like data (for STPS) '

def stats_stps(corrs1, corrs2, fisherz=True, permutation=True, iter=1000):

    """
    Conduct the statistical analysis for results of EEG-like data（for STPS）

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
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs1)) != 3 or len(np.shape(corrs2)) != 3 or np.shape(corrs1)[1] != np.shape(corrs2)[1] or \
            np.shape(corrs1)[2] != np.shape(corrs2)[2]:

        return "Invalid input!"

    # get the number of subjects, channels & time-points
    subs, chls, ts = np.shape(corrs1)

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # initialize the corrs
    stats = np.zeros([chls, ts, 2])

    # get r-map
    rs1 = corrs1
    rs2 = corrs2

    if fisherz == True:
        # Fisher r to z
        rs1 = 0.5 * np.log((1 + rs1) / (1 - rs1))
        rs2 = 0.5 * np.log((1 + rs2) / (1 - rs2))

    # calculate the statistical results
    for i in range(chls):
        for j in range(ts):

            # t test
            stats[i, j] = ttest_rel(rs1[:, i, j], rs2[:, i, j])

            if permutation == True:

                stats[i, j, 1] = permutation_test(rs1[:, i, j], rs2[:, i, j], iter=iter)

    return stats


' a function for conducting the statistical analysis for results of fMRI data (STPS searchlight) '

def stats_stpsfmri(corrs1, corrs2, fisherz=True, permutation=False, iter=1000):

    """
    Conduct the statistical analysis for results of fMRI data (STPS searchlight)

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
    fisherz : bool True or False. Default is True.
        Conduct Fisher-Z transform.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 1000.
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

    if len(np.shape(corrs1)) != 4 or len(np.shape(corrs2)) != 4 or np.shape(corrs1)[1] != np.shape(corrs2)[1] \
            or np.shape(corrs1)[2] != np.shape(corrs2)[2] or np.shape(corrs1)[3] != np.shape(corrs2)[3]:

        return "Invalid input!"

    # get the number of subjects
    subs = np.shape(corrs1)[0]

    # subs>=6
    if subs < 6:
        return print("the number of subjects is too small!")

    # get the number of the calculation units in the x, y, z directions
    n_x, n_y, n_z = np.shape(corrs1)[1:]

    # initialize the corrs
    stats = np.zeros([n_x, n_y, n_z, 2])

    # get r-map
    rs1 = corrs1
    rs2 = corrs2

    if fisherz == True:
        # Fisher r to z
        rs1 = 0.5 * np.log((1 + rs1) / (1 - rs1))
        rs2 = 0.5 * np.log((1 + rs2) / (1 - rs2))

    # calculate the statistical results
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                # t test
                stats[i, j, k] = ttest_rel(rs1[:, i, j, k], rs2[:, i, j, k])

                if permutation == True:
                    stats[i, j, k, 1] = permutation_test(rs1[:, i, j, k], rs2[:, i, j, k], iter=iter)

    return stats