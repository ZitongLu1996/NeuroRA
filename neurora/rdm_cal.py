# -*- coding: utf-8 -*-

' a module for calculating the RDM based on multimode neural data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.stuff import limtozero
from tqdm import tqdm
from scipy.stats import pearsonr
from neurora.decoding import tbyt_decoding_kfold

np.seterr(divide='ignore', invalid='ignore')


def _validate_rdm_method(method):
    """
    Validate the method for RDM calculation.

    Parameters
    ----------
    method : {"correlation", "amplitude"}
        Method used to compute dissimilarities.

    Returns
    -------
    method : str
        Validated method.
    """

    valid_methods = {"correlation", "amplitude"}

    if method not in valid_methods:
        raise ValueError(
            "method should be one of {'correlation', 'amplitude'}."
        )

    return method


def _compute_rdm_from_patterns(patterns, method="correlation"):
    """
    Compute an RDM from condition-wise feature patterns.

    Parameters
    ----------
    patterns : array_like, shape [n_cons, n_features]
        Feature patterns for each condition.

    method : {"correlation", "amplitude"}, default="correlation"
        Method used to compute dissimilarities.

        - "correlation": 1 - Pearson correlation between condition patterns.
        - "amplitude": absolute difference between the mean feature amplitudes
          of two conditions.

    Returns
    -------
    rdm : ndarray, shape [n_cons, n_cons]
        Representational dissimilarity matrix.
    """

    method = _validate_rdm_method(method)

    patterns = np.asarray(patterns)

    if patterns.ndim != 2:
        raise ValueError("patterns should have shape [n_cons, n_features].")

    n_cons = patterns.shape[0]
    rdm = np.zeros((n_cons, n_cons), dtype=float)

    for i in range(n_cons):
        for j in range(i + 1, n_cons):
            x = patterns[i]
            y = patterns[j]

            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                value = np.nan

            elif method == "correlation":
                if x.size < 2 or y.size < 2:
                    value = np.nan
                else:
                    r = pearsonr(x, y)[0]
                    value = limtozero(1 - r)

            elif method == "amplitude":
                value = np.abs(np.mean(x) - np.mean(y))

            rdm[i, j] = value
            rdm[j, i] = value

    return rdm


def _validate_eeg_rdm_input(EEG_data, chl_opt, time_opt, time_win, time_step, method):
    """
    Validate inputs for eegRDM.
    """

    EEG_data = np.asarray(EEG_data)
    method = _validate_rdm_method(method)

    if EEG_data.ndim != 5:
        raise ValueError(
            "EEG_data should have shape [n_cons, n_subs, n_trials, n_chls, n_ts]."
        )

    if chl_opt not in {0, 1}:
        raise ValueError("chl_opt should be 0 or 1.")

    if time_opt not in {0, 1}:
        raise ValueError("time_opt should be 0 or 1.")

    if not isinstance(time_win, int) or time_win < 1:
        raise ValueError("time_win should be a positive integer.")

    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step should be a positive integer.")

    if time_win > EEG_data.shape[-1]:
        raise ValueError(
            "time_win should not be larger than the number of time points."
        )

    return EEG_data, method


def _validate_fmri_searchlight_params(ksize, strides, nx, ny, nz):
    """
    Validate fMRI searchlight parameters.

    Parameters
    ----------
    ksize : array_like, length 3
        Searchlight kernel size.

    strides : array_like, length 3
        Searchlight strides.

    nx, ny, nz : int
        fMRI image dimensions.

    Returns
    -------
    kx, ky, kz, sx, sy, sz : int
        Validated kernel sizes and strides.
    """

    if len(ksize) != 3 or len(strides) != 3:
        raise ValueError("ksize and strides should both have length 3.")

    kx, ky, kz = map(int, ksize)
    sx, sy, sz = map(int, strides)

    if min(kx, ky, kz) < 1:
        raise ValueError("ksize should contain positive integers.")

    if min(sx, sy, sz) < 1:
        raise ValueError("strides should contain positive integers.")

    if kx > nx or ky > ny or kz > nz:
        raise ValueError("ksize should not be larger than the fMRI image size.")

    return kx, ky, kz, sx, sy, sz


def bhvRDM(bhv_data, sub_opt=1, method="correlation"):
    """
    Calculate RDMs for behavioral data.

    Parameters
    ----------
    bhv_data : array_like, shape [n_cons, n_subs, n_trials]
        Behavioral data.

    sub_opt : {0, 1}, default=1
        Whether to return subject-level RDMs or an averaged RDM.

        - 1: calculate one RDM for each subject.
        - 0: average trials and use subjects as features.

    method : {"correlation", "amplitude"}, default="correlation"
        Method used to calculate dissimilarities when sub_opt=0.

    Returns
    -------
    rdms : ndarray
        Behavioral RDMs.

        If sub_opt=1, shape is [n_subs, n_cons, n_cons].
        If sub_opt=0, shape is [n_cons, n_cons].
    """

    bhv_data = np.asarray(bhv_data)

    if bhv_data.ndim != 3:
        raise ValueError(
            "bhv_data should have shape [n_cons, n_subs, n_trials]."
        )

    if sub_opt not in {0, 1}:
        raise ValueError("sub_opt should be 0 or 1.")

    method = _validate_rdm_method(method)

    n_cons, n_subs, _ = bhv_data.shape

    if sub_opt == 1:
        print("\nComputing RDMs")

        rdms = np.zeros((n_subs, n_cons, n_cons), dtype=float)

        for sub in tqdm(range(n_subs), desc="Calculating"):
            means = np.mean(bhv_data[:, sub, :], axis=1)

            for i in range(n_cons):
                for j in range(i + 1, n_cons):
                    value = np.abs(means[i] - means[j])
                    rdms[sub, i, j] = value
                    rdms[sub, j, i] = value

        print("\nRDMs computing finished!")

        return rdms

    print("\nComputing RDM")

    # Shape: [n_cons, n_subs]
    data = np.mean(bhv_data, axis=2)

    rdm = _compute_rdm_from_patterns(data, method=method)

    print("\nRDM computing finished!")

    return rdm


def eegRDM(EEG_data, chl_opt=0, time_opt=0, time_win=5, time_step=5, method="correlation"):
    """
    Calculate RDMs based on EEG-like data.

    Parameters
    ----------
    EEG_data : array_like, shape [n_cons, n_subs, n_trials, n_chls, n_ts]
        EEG/MEG/fNIRS/ECoG/sEEG-like data.

    chl_opt : {0, 1}, default=0
        Whether to calculate RDMs separately for each channel.

        - 0: use all channels as features.
        - 1: calculate one RDM for each channel.

    time_opt : {0, 1}, default=0
        Whether to calculate RDMs separately for each time window.

        - 0: use all time points as features.
        - 1: calculate one RDM for each time window.

    time_win : int, default=5
        Time-window size in samples. Only used when time_opt=1.

    time_step : int, default=5
        Step size between adjacent time windows. Only used when time_opt=1.

    method : {"correlation", "amplitude"}, default="correlation"
        Method used to calculate dissimilarities.

        - "correlation": 1 - Pearson correlation between condition patterns.
        - "amplitude": absolute difference between mean feature amplitudes.

    Returns
    -------
    rdms : ndarray
        EEG-like RDMs.

        If chl_opt=0 and time_opt=0:
        - shape [n_subs, n_cons, n_cons]

        If chl_opt=0 and time_opt=1:
        - shape [n_subs, n_windows, n_cons, n_cons]

        If chl_opt=1 and time_opt=0:
        - shape [n_subs, n_chls, n_cons, n_cons]

        If chl_opt=1 and time_opt=1:
        - shape [n_subs, n_chls, n_windows, n_cons, n_cons]
    """

    EEG_data, method = _validate_eeg_rdm_input(
        EEG_data=EEG_data,
        chl_opt=chl_opt,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
        method=method,
    )

    n_cons, n_subs, _, n_chls, n_ts = EEG_data.shape

    # Average trials first.
    # Shape: [n_cons, n_subs, n_chls, n_ts]
    avg_data = np.mean(EEG_data, axis=2)

    print("\nComputing RDMs")

    if time_opt == 0 and chl_opt == 0:
        rdms = np.zeros((n_subs, n_cons, n_cons), dtype=float)

        for sub in tqdm(range(n_subs), desc="Calculating"):
            patterns = avg_data[:, sub].reshape(n_cons, n_chls * n_ts)
            rdms[sub] = _compute_rdm_from_patterns(patterns, method=method)

        print("\nRDMs computing finished!")

        return rdms

    if time_opt == 0 and chl_opt == 1:
        rdms = np.zeros((n_subs, n_chls, n_cons, n_cons), dtype=float)

        tasks = [
            (sub, chl)
            for sub in range(n_subs)
            for chl in range(n_chls)
        ]

        for sub, chl in tqdm(tasks, desc="Calculating"):
            patterns = avg_data[:, sub, chl, :]
            rdms[sub, chl] = _compute_rdm_from_patterns(patterns, method=method)

        print("\nRDMs computing finished!")

        return rdms

    n_windows = int((n_ts - time_win) / time_step) + 1

    if time_opt == 1 and chl_opt == 0:
        rdms = np.zeros((n_subs, n_windows, n_cons, n_cons), dtype=float)

        tasks = [
            (sub, t)
            for sub in range(n_subs)
            for t in range(n_windows)
        ]

        for sub, t in tqdm(tasks, desc="Calculating"):
            start = t * time_step
            stop = start + time_win
            patterns = avg_data[:, sub, :, start:stop].reshape(
                n_cons,
                n_chls * time_win,
            )
            rdms[sub, t] = _compute_rdm_from_patterns(patterns, method=method)

        print("\nRDMs computing finished!")

        return rdms

    # time_opt == 1 and chl_opt == 1
    rdms = np.zeros((n_subs, n_chls, n_windows, n_cons, n_cons), dtype=float)

    tasks = [
        (sub, chl, t)
        for sub in range(n_subs)
        for chl in range(n_chls)
        for t in range(n_windows)
    ]

    for sub, chl, t in tqdm(tasks, desc="Calculating"):
        start = t * time_step
        stop = start + time_win
        patterns = avg_data[:, sub, chl, start:stop]
        rdms[sub, chl, t] = _compute_rdm_from_patterns(patterns, method=method)

    print("\nRDMs computing finished!")

    return rdms


def eegRDM_bydecoding(EEG_data, time_win=5, time_step=5, navg=5, time_opt="average",
                      nfolds=5, nrepeats=2, normalization=False):
    """
    Calculate RDMs using classification-based neural decoding for EEG-like data.

    Parameters
    ----------
    EEG_data : array_like, shape [n_cons, n_subs, n_trials, n_chls, n_ts]
        EEG-like data.

    time_win : int, default=5
        Time-window size in samples.

    time_step : int, default=5
        Time-step size in samples.

    navg : int, default=5
        Number of trials used to average.

    time_opt : {"average", "features"}, default="average"
        Whether to average time points or use time points as features.

    nfolds : int, default=5
        Number of folds.

    nrepeats : int, default=2
        Number of repeated iterations.

    normalization : bool, default=False
        Whether to normalize data.

    Returns
    -------
    rdms : ndarray, shape [n_subs, n_windows, n_cons, n_cons]
        Decoding-based EEG-like RDMs.
    """

    EEG_data = np.asarray(EEG_data)

    if EEG_data.ndim != 5:
        raise ValueError(
            "EEG_data should have shape [n_cons, n_subs, n_trials, n_chls, n_ts]."
        )

    n_cons, n_subs, n_trials, _, n_ts = EEG_data.shape
    n_windows = int((n_ts - time_win) / time_step) + 1

    rdms = np.zeros((n_subs, n_windows, n_cons, n_cons), dtype=float)

    for con1 in range(n_cons):
        for con2 in range(con1):
            data = np.concatenate((EEG_data[con1], EEG_data[con2]), axis=1)

            labels = np.zeros((n_subs, 2 * n_trials), dtype=int)
            labels[:, n_trials:] = 1

            rdms[:, :, con1, con2] = tbyt_decoding_kfold(
                data,
                labels,
                n=2,
                navg=navg,
                time_opt=time_opt,
                time_win=time_win,
                time_step=time_step,
                nfolds=nfolds,
                nrepeats=nrepeats,
                normalization=normalization,
                pca=False,
            )

            rdms[:, :, con2, con1] = rdms[:, :, con1, con2]

    return rdms


def fmriRDM(fmri_data, ksize=(3, 3, 3), strides=(1, 1, 1), method="correlation"):
    """
    Calculate searchlight RDMs based on fMRI data.

    Parameters
    ----------
    fmri_data : array_like, shape [n_cons, n_subs, nx, ny, nz]
        fMRI data.

    ksize : tuple or list of int, default=(3, 3, 3)
        Searchlight kernel size along x, y, and z axes.

    strides : tuple or list of int, default=(1, 1, 1)
        Searchlight strides along x, y, and z axes.

    method : {"correlation", "amplitude"}, default="correlation"
        Method used to calculate dissimilarities.

        - "correlation": 1 - Pearson correlation between condition patterns.
        - "amplitude": absolute difference between mean voxel amplitudes.

    Returns
    -------
    subrdms : ndarray, shape [n_subs, n_x, n_y, n_z, n_cons, n_cons]
        fMRI searchlight RDMs.

    Notes
    -----
    A searchlight unit is skipped for all subjects if any voxel in that
    searchlight contains NaN for any condition or subject.
    """

    method = _validate_rdm_method(method)
    fmri_data = np.asarray(fmri_data)

    if fmri_data.ndim != 5:
        raise ValueError(
            "fmri_data should have shape [n_cons, n_subs, nx, ny, nz]."
        )

    n_cons, n_subs, nx, ny, nz = fmri_data.shape

    kx, ky, kz, sx, sy, sz = _validate_fmri_searchlight_params(
        ksize=ksize,
        strides=strides,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    subrdms = np.full(
        (n_subs, n_x, n_y, n_z, n_cons, n_cons),
        np.nan,
        dtype=float,
    )

    tasks = [
        (x, y, z)
        for x in range(n_x)
        for y in range(n_y)
        for z in range(n_z)
    ]

    print("\nComputing RDMs")

    for x, y, z in tqdm(tasks, desc="Calculating"):
        x0 = x * sx
        y0 = y * sy
        z0 = z * sz

        # Shape: [n_cons, n_subs, kx, ky, kz]
        patch = fmri_data[
            :,
            :,
            x0:x0 + kx,
            y0:y0 + ky,
            z0:z0 + kz,
        ]

        # Keep the user's intended strict rule:
        # if any condition or subject contains NaN in this searchlight,
        # skip this searchlight for all subjects.
        if not np.isfinite(patch).all():
            continue

        # Shape after selecting one subject:
        # [n_cons, kx, ky, kz] -> [n_cons, kx * ky * kz]
        for sub in range(n_subs):
            patterns = patch[:, sub].reshape(n_cons, kx * ky * kz)
            subrdms[sub, x, y, z] = _compute_rdm_from_patterns(
                patterns,
                method=method,
            )

    print("\nRDMs computing finished!")

    return subrdms


def fmriRDM_roi(fmri_data, mask_data, method="correlation"):
    """
    Calculate ROI RDMs based on fMRI data.

    Parameters
    ----------
    fmri_data : array_like, shape [n_cons, n_subs, nx, ny, nz]
        fMRI data.

    mask_data : array_like, shape [nx, ny, nz]
        ROI mask. Non-zero and finite voxels are included.

    method : {"correlation", "amplitude"}, default="correlation"
        Method used to calculate dissimilarities.

        - "correlation": 1 - Pearson correlation between condition patterns.
        - "amplitude": absolute difference between mean voxel amplitudes.

    Returns
    -------
    subrdms : ndarray, shape [n_subs, n_cons, n_cons]
        fMRI ROI RDMs.

    Notes
    -----
    A voxel is included only if it is non-zero in the mask, finite in the mask,
    and finite for all conditions and subjects in fmri_data.
    """

    method = _validate_rdm_method(method)

    fmri_data = np.asarray(fmri_data)
    mask_data = np.asarray(mask_data)

    if fmri_data.ndim != 5:
        raise ValueError(
            "fmri_data should have shape [n_cons, n_subs, nx, ny, nz]."
        )

    if mask_data.ndim != 3:
        raise ValueError("mask_data should have shape [nx, ny, nz].")

    n_cons, n_subs, nx, ny, nz = fmri_data.shape

    if mask_data.shape != (nx, ny, nz):
        raise ValueError(
            "The spatial shape of mask_data should match fmri_data."
        )

    # Keep the user's intended strict rule:
    # a voxel is valid only if it is finite across all conditions and subjects.
    valid_mask = (
        (mask_data != 0)
        & np.isfinite(mask_data)
        & np.isfinite(fmri_data).all(axis=(0, 1))
    )

    n_voxels = int(np.sum(valid_mask))

    if n_voxels == 0:
        raise ValueError(
            "No valid voxels found in the ROI mask after excluding zeros and NaNs."
        )

    # Shape: [n_cons, n_subs, n_voxels]
    roi_data = fmri_data[:, :, valid_mask]

    # Shape: [n_subs, n_cons, n_voxels]
    roi_data = np.transpose(roi_data, (1, 0, 2))

    subrdms = np.full((n_subs, n_cons, n_cons), np.nan, dtype=float)

    print("\nComputing RDMs")

    for sub in tqdm(range(n_subs), desc="Calculating"):
        patterns = roi_data[sub]
        subrdms[sub] = _compute_rdm_from_patterns(patterns, method=method)

    print("\nRDMs computing finished!")

    return subrdms