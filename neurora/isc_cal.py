# -*- coding: utf-8 -*-

' a module for calculating the inter-subject correlation based on neural data '

__author__ = 'Zitong Lu'

import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm
from neurora.stuff import _validate_parallel_args
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')


def _pearsonr_safe(x, y):
    """
    Safely compute Pearson correlation.

    Parameters
    ----------
    x : array_like
        First vector.

    y : array_like
        Second vector.

    Returns
    -------
    rp : ndarray, shape [2]
        Pearson correlation result.

        - rp[0] : r-value
        - rp[1] : p-value
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y should have the same shape.")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y should be one-dimensional vectors.")

    if x.size < 2:
        return np.array([np.nan, np.nan], dtype=float)

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.array([np.nan, np.nan], dtype=float)

    # Pearson correlation is undefined for constant vectors.
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.array([np.nan, np.nan], dtype=float)

    r, p = pearsonr(x, y)

    return np.array([r, p], dtype=float)


def _get_subject_pairs(n_subs):
    """
    Get all unique subject pairs.

    Parameters
    ----------
    n_subs : int
        Number of subjects.

    Returns
    -------
    subject_pairs : list of tuple
        List of subject index pairs.
    """

    if n_subs < 2:
        raise ValueError("At least two subjects are required to compute ISC.")

    return list(combinations(range(n_subs), 2))


def _validate_time_window(n_ts, time_win, time_step):
    """
    Validate time-window parameters and return number of windows.

    Parameters
    ----------
    n_ts : int
        Number of time points.

    time_win : int
        Time-window size.

    time_step : int
        Time-step size.

    Returns
    -------
    n_windows : int
        Number of time windows.
    """

    if not isinstance(time_win, int) or time_win < 1:
        raise ValueError("time_win should be a positive integer.")

    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step should be a positive integer.")

    if time_win > n_ts:
        raise ValueError("time_win should not be larger than the number of time points.")

    n_windows = int((n_ts - time_win) / time_step) + 1

    return n_windows


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


def _compute_isc_single_task(data, pair, chl, time_idx, time_win, time_step):
    """
    Compute ISC for one subject pair, one channel, and one time window.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_chls, n_ts]
        EEG-like data.

    pair : tuple of int
        Subject pair.

    chl : int
        Channel index.

    time_idx : int
        Time-window index.

    time_win : int
        Time-window size.

    time_step : int
        Time-step size.

    Returns
    -------
    rp : ndarray, shape [2]
        Pearson correlation result.
    """

    sub1, sub2 = pair
    start = time_idx * time_step
    stop = start + time_win

    x = data[sub1, chl, start:stop]
    y = data[sub2, chl, start:stop]

    return _pearsonr_safe(x, y)


def _compute_fmri_isc_single_task(fmri_data, pair, t, x, y, z, kx, ky, kz, sx, sy, sz):
    """
    Compute fMRI searchlight ISC for one time point, one subject pair,
    and one searchlight unit.

    Parameters
    ----------
    fmri_data : ndarray, shape [n_ts, n_subs, nx, ny, nz]
        fMRI data.

    pair : tuple of int
        Subject pair.

    t : int
        Time-point index.

    x, y, z : int
        Searchlight indices.

    kx, ky, kz : int
        Kernel sizes.

    sx, sy, sz : int
        Strides.

    Returns
    -------
    rp : ndarray, shape [2]
        Pearson correlation result.
    """

    sub1, sub2 = pair

    x0 = x * sx
    y0 = y * sy
    z0 = z * sz

    patch1 = fmri_data[
        t,
        sub1,
        x0:x0 + kx,
        y0:y0 + ky,
        z0:z0 + kz,
    ]

    patch2 = fmri_data[
        t,
        sub2,
        x0:x0 + kx,
        y0:y0 + ky,
        z0:z0 + kz,
    ]

    return _pearsonr_safe(
        patch1.reshape(kx * ky * kz),
        patch2.reshape(kx * ky * kz),
    )


def _compute_fmri_roi_isc_single_task(roi_data, pair, t):
    """
    Compute fMRI ROI ISC for one time point and one subject pair.

    Parameters
    ----------
    roi_data : ndarray, shape [n_ts, n_subs, n_voxels]
        ROI data.

    pair : tuple of int
        Subject pair.

    t : int
        Time-point index.

    Returns
    -------
    rp : ndarray, shape [2]
        Pearson correlation result.
    """

    sub1, sub2 = pair

    return _pearsonr_safe(
        roi_data[t, sub1],
        roi_data[t, sub2],
    )


def isc(data, time_win=5, time_step=5, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate inter-subject correlation (ISC) for EEG-like data.

    Parameters
    ----------
    data : array_like, shape [n_subs, n_chls, n_ts]
        EEG-like neural data.

        n_subs, n_chls, and n_ts represent the number of subjects, channels,
        and time points, respectively.

    time_win : int, default=5
        Time-window size in samples.

    time_step : int, default=5
        Step size between adjacent time windows.

    parallel : bool, default=False
        Whether to parallelize across subject pair × channel × time-window tasks.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    isc_results : ndarray, shape [n_pairs, n_chls, n_windows, 2]
        ISC results.

        The last dimension contains:

        - isc_results[..., 0] : r-value
        - isc_results[..., 1] : p-value

    Notes
    -----
    ISC is computed for each unique pair of subjects.
    """

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("data should have shape [n_subs, n_chls, n_ts].")

    n_subs, n_chls, n_ts = data.shape
    n_windows = _validate_time_window(n_ts, time_win, time_step)
    n_jobs, backend = _validate_parallel_args(parallel, n_jobs, backend)

    subject_pairs = _get_subject_pairs(n_subs)
    n_pairs = len(subject_pairs)

    isc_results = np.full(
        (n_pairs, n_chls, n_windows, 2),
        np.nan,
        dtype=float,
    )

    tasks = [
        (pair_idx, pair, chl, time_idx)
        for pair_idx, pair in enumerate(subject_pairs)
        for chl in range(n_chls)
        for time_idx in range(n_windows)
    ]

    print("\nISC starts")

    if parallel:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_compute_isc_single_task)(
                data,
                pair,
                chl,
                time_idx,
                time_win,
                time_step,
            )
            for _, pair, chl, time_idx in tqdm(tasks, desc="Calculating")
        )

        for (pair_idx, _, chl, time_idx), rp in zip(tasks, results):
            isc_results[pair_idx, chl, time_idx] = rp

    else:
        for pair_idx, pair, chl, time_idx in tqdm(tasks, desc="Calculating"):
            isc_results[pair_idx, chl, time_idx] = _compute_isc_single_task(
                data=data,
                pair=pair,
                chl=chl,
                time_idx=time_idx,
                time_win=time_win,
                time_step=time_step,
            )

    print("\nComputing finished!")

    return isc_results


def isc_fmri(fmri_data, ksize=(3, 3, 3), strides=(1, 1, 1), parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate inter-subject correlation (ISC) for fMRI data using searchlights.

    Parameters
    ----------
    fmri_data : array_like, shape [n_ts, n_subs, nx, ny, nz]
        fMRI data.

        n_ts, n_subs, nx, ny, and nz represent the number of time points,
        subjects, and spatial dimensions.

    ksize : tuple or list of int, default=(3, 3, 3)
        Searchlight kernel size along x, y, and z axes.

    strides : tuple or list of int, default=(1, 1, 1)
        Searchlight strides along x, y, and z axes.

    parallel : bool, default=False
        Whether to parallelize across time point × subject pair × searchlight tasks.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    isc_results : ndarray, shape [n_ts, n_pairs, n_x, n_y, n_z, 2]
        fMRI searchlight ISC results.

        The last dimension contains:

        - isc_results[..., 0] : r-value
        - isc_results[..., 1] : p-value

    Notes
    -----
    ISC is computed for each unique pair of subjects.

    The NaN handling follows the original logic: for each time point, subject
    pair, and searchlight unit, the ISC is set to NaN if either subject's
    searchlight vector contains NaN.
    """

    fmri_data = np.asarray(fmri_data)

    if fmri_data.ndim != 5:
        raise ValueError(
            "fmri_data should have shape [n_ts, n_subs, nx, ny, nz]."
        )

    n_ts, n_subs, nx, ny, nz = fmri_data.shape

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

    n_jobs, backend = _validate_parallel_args(parallel, n_jobs, backend)

    subject_pairs = _get_subject_pairs(n_subs)
    n_pairs = len(subject_pairs)

    isc_results = np.full(
        (n_ts, n_pairs, n_x, n_y, n_z, 2),
        np.nan,
        dtype=float,
    )

    tasks = [
        (t, pair_idx, pair, x, y, z)
        for t in range(n_ts)
        for pair_idx, pair in enumerate(subject_pairs)
        for x in range(n_x)
        for y in range(n_y)
        for z in range(n_z)
    ]

    print("\nISC starts")

    if parallel:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_compute_fmri_isc_single_task)(
                fmri_data,
                pair,
                t,
                x,
                y,
                z,
                kx,
                ky,
                kz,
                sx,
                sy,
                sz,
            )
            for t, _, pair, x, y, z in tqdm(tasks, desc="Calculating")
        )

        for (t, pair_idx, _, x, y, z), rp in zip(tasks, results):
            isc_results[t, pair_idx, x, y, z] = rp

    else:
        for t, pair_idx, pair, x, y, z in tqdm(tasks, desc="Calculating"):
            isc_results[t, pair_idx, x, y, z] = _compute_fmri_isc_single_task(
                fmri_data=fmri_data,
                pair=pair,
                t=t,
                x=x,
                y=y,
                z=z,
                kx=kx,
                ky=ky,
                kz=kz,
                sx=sx,
                sy=sy,
                sz=sz,
            )

    print("\nComputing finished!")

    return isc_results


def isc_fmri_roi(fmri_data, mask_data, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate inter-subject correlation (ISC) for fMRI data within an ROI.

    Parameters
    ----------
    fmri_data : array_like, shape [n_ts, n_subs, nx, ny, nz]
        fMRI data.

    mask_data : array_like, shape [nx, ny, nz]
        ROI mask. Non-zero and finite voxels are included.

    parallel : bool, default=False
        Whether to parallelize across time point × subject pair tasks.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    isc_results : ndarray, shape [n_ts, n_pairs, 2]
        fMRI ROI ISC results.

        The last dimension contains:

        - isc_results[..., 0] : r-value
        - isc_results[..., 1] : p-value

    Notes
    -----
    ISC is computed for each unique pair of subjects.

    The ROI voxel selection follows the original logic: voxels are included
    if the mask is non-zero and finite. fMRI NaNs are handled at the
    time point × subject pair level, where the corresponding ISC value
    becomes NaN if either subject's ROI vector contains NaN.
    """

    fmri_data = np.asarray(fmri_data)
    mask_data = np.asarray(mask_data)

    if fmri_data.ndim != 5:
        raise ValueError(
            "fmri_data should have shape [n_ts, n_subs, nx, ny, nz]."
        )

    if mask_data.ndim != 3:
        raise ValueError("mask_data should have shape [nx, ny, nz].")

    n_ts, n_subs, nx, ny, nz = fmri_data.shape

    if mask_data.shape != (nx, ny, nz):
        raise ValueError(
            "The spatial shape of mask_data should match fmri_data."
        )

    valid_mask = (mask_data != 0) & np.isfinite(mask_data)
    n_voxels = int(np.sum(valid_mask))

    if n_voxels == 0:
        raise ValueError(
            "No valid voxels found in the ROI mask after excluding zeros and NaNs."
        )

    # Shape: [n_ts, n_subs, n_voxels]
    roi_data = fmri_data[:, :, valid_mask]

    n_jobs, backend = _validate_parallel_args(parallel, n_jobs, backend)

    subject_pairs = _get_subject_pairs(n_subs)
    n_pairs = len(subject_pairs)

    isc_results = np.full((n_ts, n_pairs, 2), np.nan, dtype=float)

    tasks = [
        (t, pair_idx, pair)
        for t in range(n_ts)
        for pair_idx, pair in enumerate(subject_pairs)
    ]

    print("\nISC starts")

    if parallel:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_compute_fmri_roi_isc_single_task)(
                roi_data,
                pair,
                t,
            )
            for t, _, pair in tqdm(tasks, desc="Calculating")
        )

        for (t, pair_idx, _), rp in zip(tasks, results):
            isc_results[t, pair_idx] = rp

    else:
        for t, pair_idx, pair in tqdm(tasks, desc="Calculating"):
            isc_results[t, pair_idx] = _compute_fmri_roi_isc_single_task(
                roi_data=roi_data,
                pair=pair,
                t=t,
            )

    print("\nComputing finished!")

    return isc_results