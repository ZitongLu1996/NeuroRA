# -*- coding: utf-8 -*-

"""
A module for calculating cross-temporal RDMs based on EEG-like data.
"""

__author__ = "Zitong Lu"

import numpy as np
from tqdm import tqdm


def _validate_eeg_data(data):
    """
    Validate EEG-like data for cross-temporal RDM calculation.

    Parameters
    ----------
    data : array_like, shape [n_cons, n_subs, n_chls, n_ts]
        EEG-like data.

    Returns
    -------
    data : ndarray, shape [n_cons, n_subs, n_chls, n_ts]
        Validated data.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 4:
        raise ValueError(
            "data should have shape [n_cons, n_subs, n_chls, n_ts]."
        )

    n_cons, n_subs, n_chls, n_ts = data.shape

    if n_cons < 2:
        raise ValueError("data should contain at least two conditions.")

    if n_subs < 1:
        raise ValueError("data should contain at least one subject.")

    if n_chls < 1:
        raise ValueError("data should contain at least one channel.")

    if n_ts < 1:
        raise ValueError("data should contain at least one time point.")

    return data


def _validate_ctrdm_params(chl_opt, time_win, time_step, n_ts):
    """
    Validate parameters for cross-temporal RDM calculation.

    Parameters
    ----------
    chl_opt : int
        Whether to calculate channel-specific CTRDMs.

    time_win : int
        Time window size.

    time_step : int
        Time step size.

    n_ts : int
        Number of time points.

    Returns
    -------
    chl_opt : int
        Validated chl_opt.

    time_win : int
        Validated time_win.

    time_step : int
        Validated time_step.

    n_windows : int
        Number of time windows.
    """

    if chl_opt not in {0, 1}:
        raise ValueError("chl_opt should be 0 or 1.")

    if not isinstance(time_win, int) or time_win < 1:
        raise ValueError("time_win should be a positive integer.")

    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step should be a positive integer.")

    if time_win > n_ts:
        raise ValueError("time_win should not be larger than n_ts.")

    n_windows = int((n_ts - time_win) / time_step) + 1

    return chl_opt, time_win, time_step, n_windows


def _extract_time_windows(data, time_win, time_step):
    """
    Extract sliding time windows from EEG-like data.

    Parameters
    ----------
    data : ndarray, shape [n_cons, n_subs, n_chls, n_ts]
        EEG-like data.

    time_win : int
        Time window size.

    time_step : int
        Time step size.

    Returns
    -------
    windowed_data : ndarray, shape [n_cons, n_subs, n_windows, n_chls, time_win]
        Windowed data.
    """

    n_cons, n_subs, n_chls, n_ts = data.shape
    n_windows = int((n_ts - time_win) / time_step) + 1

    windowed_data = np.zeros(
        (n_cons, n_subs, n_windows, n_chls, time_win),
        dtype=float,
    )

    for t in range(n_windows):
        start = t * time_step
        stop = start + time_win
        windowed_data[:, :, t] = data[:, :, :, start:stop]

    return windowed_data


def _pairwise_correlation_distance(patterns_x, patterns_y):
    """
    Calculate a correlation-distance matrix between two sets of condition patterns.

    Parameters
    ----------
    patterns_x : ndarray, shape [n_cons, n_features]
        Condition patterns from one time window.

    patterns_y : ndarray, shape [n_cons, n_features]
        Condition patterns from another time window.

    Returns
    -------
    rdm : ndarray, shape [n_cons, n_cons]
        Correlation-distance matrix, defined as 1 - Pearson r.

        Diagonal values are set to 0.
    """

    patterns_x = np.asarray(patterns_x, dtype=float)
    patterns_y = np.asarray(patterns_y, dtype=float)

    if patterns_x.shape != patterns_y.shape:
        raise ValueError("patterns_x and patterns_y should have the same shape.")

    n_cons, n_features = patterns_x.shape

    if n_features < 2:
        rdm = np.full((n_cons, n_cons), np.nan, dtype=float)
        np.fill_diagonal(rdm, 0)
        return rdm

    x = patterns_x - np.nanmean(patterns_x, axis=1, keepdims=True)
    y = patterns_y - np.nanmean(patterns_y, axis=1, keepdims=True)

    x_norm = np.sqrt(np.nansum(x ** 2, axis=1, keepdims=True))
    y_norm = np.sqrt(np.nansum(y ** 2, axis=1, keepdims=True))

    denom = x_norm @ y_norm.T
    corr = (x @ y.T) / denom

    corr[denom == 0] = np.nan

    rdm = 1 - corr
    np.fill_diagonal(rdm, 0)

    return rdm


def ctRDM(data, chl_opt=0, time_win=5, time_step=5):
    """
    Calculate cross-temporal RDMs for EEG-like data.

    Parameters
    ----------
    data : array_like, shape [n_cons, n_subs, n_chls, n_ts]
        EEG-like data.

        n_cons, n_subs, n_chls, and n_ts represent the number of conditions,
        subjects, channels, and time points, respectively.

    chl_opt : int, default=0
        Whether to calculate channel-specific CTRDMs.

        - 0 : calculate CTRDMs after combining channels and time points
        - 1 : calculate CTRDMs separately for each channel

    time_win : int, default=5
        Time-window size for calculating CTRDMs.

    time_step : int, default=5
        Step size between adjacent time windows.

    Returns
    -------
    ctrdms : ndarray
        Cross-temporal RDMs.

        If chl_opt=0, the shape is:

        [n_subs, n_windows, n_windows, n_cons, n_cons]

        If chl_opt=1, the shape is:

        [n_subs, n_chls, n_windows, n_windows, n_cons, n_cons]

    Notes
    -----
    This function always returns subject-level CTRDMs.

    If you want the group-averaged CTRDMs, use:

    np.nanmean(ctrdms, axis=0)
    """

    data = _validate_eeg_data(data)

    n_cons, n_subs, n_chls, n_ts = data.shape

    chl_opt, time_win, time_step, n_windows = _validate_ctrdm_params(
        chl_opt=chl_opt,
        time_win=time_win,
        time_step=time_step,
        n_ts=n_ts,
    )

    windowed_data = _extract_time_windows(
        data=data,
        time_win=time_win,
        time_step=time_step,
    )

    if chl_opt == 0:
        # Combine channels and time points as features.
        # Shape: [n_cons, n_subs, n_windows, n_chls * time_win]
        patterns = np.reshape(
            windowed_data,
            (n_cons, n_subs, n_windows, n_chls * time_win),
        )

        ctrdms = np.zeros(
            (n_subs, n_windows, n_windows, n_cons, n_cons),
            dtype=float,
        )

        iterator = tqdm(
            range(n_subs * n_windows * n_windows),
            desc="Calculating CTRDMs",
        )

        for index in iterator:
            sub = index // (n_windows * n_windows)
            rem = index % (n_windows * n_windows)
            t1 = rem // n_windows
            t2 = rem % n_windows

            ctrdms[sub, t1, t2] = _pairwise_correlation_distance(
                patterns[:, sub, t1],
                patterns[:, sub, t2],
            )

    else:
        # Calculate CTRDMs separately for each channel.
        # Shape used per channel: [n_cons, time_win]
        ctrdms = np.zeros(
            (n_subs, n_chls, n_windows, n_windows, n_cons, n_cons),
            dtype=float,
        )

        iterator = tqdm(
            range(n_subs * n_chls * n_windows * n_windows),
            desc="Calculating CTRDMs",
        )

        for index in iterator:
            sub = index // (n_chls * n_windows * n_windows)
            rem = index % (n_chls * n_windows * n_windows)

            chl = rem // (n_windows * n_windows)
            rem = rem % (n_windows * n_windows)

            t1 = rem // n_windows
            t2 = rem % n_windows

            ctrdms[sub, chl, t1, t2] = _pairwise_correlation_distance(
                windowed_data[:, sub, t1, chl],
                windowed_data[:, sub, t2, chl],
            )

    print("\nCross-temporal RDMs computing finished!")

    return ctrdms