# -*- coding: utf-8 -*-

"""
A module for calculating cross-temporal similarities.
"""

__author__ = "Zitong Lu"

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from neurora.rdm_corr import rdm_correlation_spearman
from neurora.ctrdm_corr import ctrdm_correlation_spearman


def _validate_eeg_pair(data1, data2):
    """
    Validate two EEG-like data arrays.

    Parameters
    ----------
    data1, data2 : array_like, shape [n_subs, n_chls, n_ts]
        EEG-like data under two conditions.

    Returns
    -------
    data1, data2 : ndarray
        Validated data arrays.
    """

    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)

    if data1.ndim != 3:
        raise ValueError("data1 should have shape [n_subs, n_chls, n_ts].")

    if data2.ndim != 3:
        raise ValueError("data2 should have shape [n_subs, n_chls, n_ts].")

    if data1.shape != data2.shape:
        raise ValueError("data1 and data2 should have the same shape.")

    n_subs, n_chls, n_ts = data1.shape

    if n_subs < 1:
        raise ValueError("data should contain at least one subject.")

    if n_chls < 1:
        raise ValueError("data should contain at least one channel.")

    if n_ts < 1:
        raise ValueError("data should contain at least one time point.")

    return data1, data2


def _validate_ctsim_params(chl_opt, time_win, time_step, n_ts):
    """
    Validate parameters for cross-temporal similarity calculation.
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


def _extract_time_windows_3d(data, time_win, time_step):
    """
    Extract sliding time windows from EEG-like data.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_chls, n_ts]
        EEG-like data.

    time_win : int
        Time-window size.

    time_step : int
        Step size between adjacent windows.

    Returns
    -------
    windowed_data : ndarray, shape [n_subs, n_windows, n_chls, time_win]
        Windowed data.
    """

    n_subs, n_chls, n_ts = data.shape
    n_windows = int((n_ts - time_win) / time_step) + 1

    windowed_data = np.zeros(
        (n_subs, n_windows, n_chls, time_win),
        dtype=float,
    )

    for t in range(n_windows):
        start = t * time_step
        stop = start + time_win
        windowed_data[:, t] = data[:, :, start:stop]

    return windowed_data


def _spearman_corr_vector(x, y):
    """
    Calculate Spearman correlation between two vectors.

    Parameters
    ----------
    x, y : array_like
        Input vectors.

    Returns
    -------
    corr : ndarray, shape [2]
        [r, p].
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y should have the same shape.")

    valid_mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid_mask) < 2:
        return np.array([np.nan, np.nan])

    r, p = spearmanr(x[valid_mask], y[valid_mask])

    return np.array([r, p])


def _validate_temporal_rdms(rdms):
    """
    Validate temporal RDMs.

    Parameters
    ----------
    rdms : array_like, shape [..., n_ts, n_cons, n_cons]
        Temporal RDMs.

    Returns
    -------
    rdms : ndarray
        Validated temporal RDMs.
    """

    rdms = np.asarray(rdms, dtype=float)

    if rdms.ndim < 3:
        raise ValueError(
            "rdms should have shape [n_ts, n_cons, n_cons] "
            "or [..., n_ts, n_cons, n_cons]."
        )

    if rdms.shape[-1] != rdms.shape[-2]:
        raise ValueError("The last two dimensions of rdms should be square.")

    if rdms.shape[-1] < 2:
        raise ValueError("RDMs should contain at least two conditions.")

    return rdms


def _validate_ctrdms_for_model(ctrdms, model_rdm):
    """
    Validate CTRDMs and model RDM.

    Parameters
    ----------
    ctrdms : array_like, shape [..., n_ts, n_ts, n_cons, n_cons]
        Cross-temporal RDMs.

    model_rdm : array_like, shape [n_cons, n_cons]
        Model RDM.

    Returns
    -------
    ctrdms : ndarray
        Validated CTRDMs.

    model_rdm : ndarray
        Validated model RDM.
    """

    ctrdms = np.asarray(ctrdms, dtype=float)
    model_rdm = np.asarray(model_rdm, dtype=float)

    if ctrdms.ndim < 4:
        raise ValueError(
            "ctrdms should have shape [n_ts, n_ts, n_cons, n_cons] "
            "or [..., n_ts, n_ts, n_cons, n_cons]."
        )

    if ctrdms.shape[-1] != ctrdms.shape[-2]:
        raise ValueError("The last two dimensions of ctrdms should be square.")

    if ctrdms.shape[-4] != ctrdms.shape[-3]:
        raise ValueError("The two time dimensions of ctrdms should have the same size.")

    if model_rdm.ndim != 2:
        raise ValueError("model_rdm should have shape [n_cons, n_cons].")

    if model_rdm.shape[0] != model_rdm.shape[1]:
        raise ValueError("model_rdm should be a square matrix.")

    if model_rdm.shape != ctrdms.shape[-2:]:
        raise ValueError(
            "model_rdm should have the same condition dimensions as ctrdms."
        )

    return ctrdms, model_rdm


def ctsim_cal(data1, data2, chl_opt=0, time_win=5, time_step=5):
    """
    Calculate cross-temporal similarities between neural data under two conditions.

    Parameters
    ----------
    data1, data2 : array_like, shape [n_subs, n_chls, n_ts]
        EEG/MEG/fNIRS-like data under two conditions.

    chl_opt : int, default=0
        Whether to calculate channel-specific cross-temporal similarities.

        - 0 : combine channels and time points as features
        - 1 : calculate similarities separately for each channel

    time_win : int, default=5
        Time-window size.

    time_step : int, default=5
        Step size between adjacent time windows.

    Returns
    -------
    ctsimilarities : ndarray
        Cross-temporal similarities.

        If chl_opt=0, shape is:

        [n_subs, n_windows, n_windows, 2]

        If chl_opt=1, shape is:

        [n_subs, n_chls, n_windows, n_windows, 2]

        The last dimension contains [r, p].
    """

    data1, data2 = _validate_eeg_pair(data1, data2)

    n_subs, n_chls, n_ts = data1.shape

    chl_opt, time_win, time_step, n_windows = _validate_ctsim_params(
        chl_opt=chl_opt,
        time_win=time_win,
        time_step=time_step,
        n_ts=n_ts,
    )

    windowed_data1 = _extract_time_windows_3d(
        data=data1,
        time_win=time_win,
        time_step=time_step,
    )

    windowed_data2 = _extract_time_windows_3d(
        data=data2,
        time_win=time_win,
        time_step=time_step,
    )

    if chl_opt == 0:
        # Combine channels and time points as features.
        # Shape: [n_subs, n_windows, n_chls * time_win]
        patterns1 = np.reshape(
            windowed_data1,
            (n_subs, n_windows, n_chls * time_win),
        )
        patterns2 = np.reshape(
            windowed_data2,
            (n_subs, n_windows, n_chls * time_win),
        )

        ctsimilarities = np.zeros(
            (n_subs, n_windows, n_windows, 2),
            dtype=float,
        )

        iterator = tqdm(
            range(n_subs * n_windows * n_windows),
            desc="Calculating cross-temporal similarities",
        )

        for index in iterator:
            sub = index // (n_windows * n_windows)
            rem = index % (n_windows * n_windows)
            t1 = rem // n_windows
            t2 = rem % n_windows

            ctsimilarities[sub, t1, t2] = _spearman_corr_vector(
                patterns1[sub, t1],
                patterns2[sub, t2],
            )

    else:
        # Calculate similarities separately for each channel.
        # Shape per channel: [time_win]
        ctsimilarities = np.zeros(
            (n_subs, n_chls, n_windows, n_windows, 2),
            dtype=float,
        )

        iterator = tqdm(
            range(n_subs * n_chls * n_windows * n_windows),
            desc="Calculating cross-temporal similarities",
        )

        for index in iterator:
            sub = index // (n_chls * n_windows * n_windows)
            rem = index % (n_chls * n_windows * n_windows)

            chl = rem // (n_windows * n_windows)
            rem = rem % (n_windows * n_windows)

            t1 = rem // n_windows
            t2 = rem % n_windows

            ctsimilarities[sub, chl, t1, t2] = _spearman_corr_vector(
                windowed_data1[sub, t1, chl],
                windowed_data2[sub, t2, chl],
            )

    print("\nCross-temporal similarities computing finished!")

    return ctsimilarities


def ctsim_rdms_cal(rdms):
    """
    Calculate cross-temporal similarities based on temporal RDMs.

    Parameters
    ----------
    rdms : array_like, shape [n_ts, n_cons, n_cons] or [..., n_ts, n_cons, n_cons]
        Temporal RDMs.

        The leading dimensions can represent subjects, channels, ROIs, etc.

    Returns
    -------
    ctsimilarities : ndarray
        Cross-temporal similarities.

        If rdms has shape [n_ts, n_cons, n_cons], output shape is:

        [n_ts, n_ts, 2]

        If rdms has shape [..., n_ts, n_cons, n_cons], output shape is:

        [..., n_ts, n_ts, 2]

        The last dimension contains [r, p].
    """

    rdms = _validate_temporal_rdms(rdms)

    leading_shape = rdms.shape[:-3]
    n_ts = rdms.shape[-3]
    n_cons = rdms.shape[-1]

    rdms_reshaped = np.reshape(
        rdms,
        (-1, n_ts, n_cons, n_cons),
    )

    n_leading = rdms_reshaped.shape[0]

    ctsim = np.zeros(
        (n_leading, n_ts, n_ts, 2),
        dtype=float,
    )

    iterator = tqdm(
        range(n_leading * n_ts * n_ts),
        desc="Calculating cross-temporal RDM similarities",
    )

    for index in iterator:
        lead = index // (n_ts * n_ts)
        rem = index % (n_ts * n_ts)
        t1 = rem // n_ts
        t2 = rem % n_ts

        ctsim[lead, t1, t2] = rdm_correlation_spearman(
            rdms_reshaped[lead, t1],
            rdms_reshaped[lead, t2],
        )

    if leading_shape == ():
        return ctsim[0]

    output_shape = leading_shape + (n_ts, n_ts, 2)

    return np.reshape(ctsim, output_shape)


def ctsim_ctrdms_cal(ctrdms, model_rdm):
    """
    Calculate cross-temporal similarities between CTRDMs and a model RDM.

    Parameters
    ----------
    ctrdms : array_like, shape [n_ts, n_ts, n_cons, n_cons] or [..., n_ts, n_ts, n_cons, n_cons]
        Cross-temporal RDMs.

        The leading dimensions can represent subjects, channels, ROIs, etc.

    model_rdm : array_like, shape [n_cons, n_cons]
        Model RDM.

    Returns
    -------
    ctsimilarities : ndarray
        Cross-temporal similarities.

        If ctrdms has shape [n_ts, n_ts, n_cons, n_cons], output shape is:

        [n_ts, n_ts, 2]

        If ctrdms has shape [..., n_ts, n_ts, n_cons, n_cons], output shape is:

        [..., n_ts, n_ts, 2]

        The last dimension contains [r, p].

    Notes
    -----
    CTRDMs are not assumed to be symmetric. Therefore, all off-diagonal
    elements are used when correlating each CTRDM with the model RDM.
    """

    ctrdms, model_rdm = _validate_ctrdms_for_model(
        ctrdms=ctrdms,
        model_rdm=model_rdm,
    )

    leading_shape = ctrdms.shape[:-4]
    n_ts = ctrdms.shape[-4]
    n_cons = ctrdms.shape[-1]

    ctrdms_reshaped = np.reshape(
        ctrdms,
        (-1, n_ts, n_ts, n_cons, n_cons),
    )

    n_leading = ctrdms_reshaped.shape[0]

    ctsim = np.zeros(
        (n_leading, n_ts, n_ts, 2),
        dtype=float,
    )

    iterator = tqdm(
        range(n_leading * n_ts * n_ts),
        desc="Calculating CTRDM-model similarities",
    )

    for index in iterator:
        lead = index // (n_ts * n_ts)
        rem = index % (n_ts * n_ts)
        t1 = rem // n_ts
        t2 = rem % n_ts

        ctsim[lead, t1, t2] = ctrdm_correlation_spearman(
            ctrdms_reshaped[lead, t1, t2],
            model_rdm,
        )

    if leading_shape == ():
        return ctsim[0]

    output_shape = leading_shape + (n_ts, n_ts, 2)

    return np.reshape(ctsim, output_shape)