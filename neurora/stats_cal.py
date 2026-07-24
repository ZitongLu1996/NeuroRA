# -*- coding: utf-8 -*-

' a module for conducting the statistical analysis '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import ttest_1samp

np.seterr(divide="ignore", invalid="ignore")

_MIN_SUBS = 6
_MIN_PAIRS = 6


def _validate_alternative(alternative):
    """
    Validate the alternative hypothesis for t-tests.

    Parameters
    ----------
    alternative : {"two-sided", "less", "greater"}
        Alternative hypothesis.

    Returns
    -------
    alternative : str
        Validated alternative hypothesis.
    """

    valid_alternatives = {"two-sided", "less", "greater"}

    if alternative not in valid_alternatives:
        raise ValueError(
            "alternative should be one of {'two-sided', 'less', 'greater'}."
        )

    return alternative


def _validate_r_values(corrs, expected_ndim, data_name="corrs"):
    """
    Validate r-value array.

    Parameters
    ----------
    corrs : array_like
        Correlation coefficients.

    expected_ndim : int
        Expected number of dimensions.

    data_name : str, default="corrs"
        Name used in error messages.

    Returns
    -------
    rs : ndarray
        Validated r-values.
    """

    rs = np.asarray(corrs, dtype=float)

    if rs.ndim != expected_ndim:
        raise ValueError(
            f"{data_name} should have {expected_ndim} dimensions."
        )

    return rs


def _fisher_z_transform(rs):
    """
    Apply Fisher z transformation to correlation coefficients.

    Parameters
    ----------
    rs : array_like
        Correlation coefficients.

    Returns
    -------
    zs : ndarray
        Fisher z-transformed values.
    """

    rs = np.asarray(rs, dtype=float)

    # Avoid infinite values when r is exactly -1 or 1.
    eps = np.finfo(float).eps
    rs = np.clip(rs, -1 + eps, 1 - eps)

    return np.arctanh(rs)


def _ttest_1samp_along_axis(
    data,
    axis=0,
    popmean=0,
    alternative="greater",
    min_n=_MIN_SUBS,
):
    """
    Conduct one-sample t-tests along a given axis.

    Parameters
    ----------
    data : ndarray
        Input data.

    axis : int, default=0
        Axis along which to perform one-sample t-tests.

    popmean : float, default=0
        Population mean.

    alternative : {"two-sided", "less", "greater"}, default="greater"
        Alternative hypothesis.

    min_n : int, default=MIN_SUBS
        Minimum number of finite observations required for each test.

    Returns
    -------
    stat_results : ndarray
        Statistical results.

        The shape is data.shape without the tested axis plus a final
        dimension of length 2.

        The last dimension contains:

        - stat_results[..., 0] : t-value
        - stat_results[..., 1] : p-value
    """

    alternative = _validate_alternative(alternative)
    data = np.asarray(data, dtype=float)

    finite_counts = np.sum(np.isfinite(data), axis=axis)

    t_values, p_values = ttest_1samp(
        data,
        popmean=popmean,
        axis=axis,
        nan_policy="omit",
        alternative=alternative,
    )

    t_values = np.asarray(t_values, dtype=float)
    p_values = np.asarray(p_values, dtype=float)

    invalid = finite_counts < min_n
    t_values[invalid] = np.nan
    p_values[invalid] = np.nan

    stat_results = np.stack((t_values, p_values), axis=-1)

    return stat_results


def stats(corrs, alternative="greater"):
    """
    Conduct statistical analysis for EEG-like correlation results.

    Parameters
    ----------
    corrs : array_like, shape [n_subs, n_chls, n_ts]
        Correlation coefficients.

        n_subs, n_chls, and n_ts represent the number of subjects, channels,
        and time points, respectively.

    alternative : {"two-sided", "less", "greater"}, default="greater"
        Alternative hypothesis for one-sample t-tests against zero.

    Returns
    -------
    stat_results : ndarray, shape [n_chls, n_ts, 2]
        Statistical results.

        The last dimension contains:

        - stat_results[..., 0] : t-value
        - stat_results[..., 1] : p-value

    Notes
    -----
    n_subs must be at least 6.

    Fisher z transformation is always applied before the t-test.

    This function can be used for subject-level EEG-like correlation results,
    such as NPS, ISC summaries, or EEG-like RDM correlations.
    """

    rs = _validate_r_values(
        corrs,
        expected_ndim=3,
        data_name="corrs",
    )

    if rs.shape[0] < _MIN_SUBS:
        raise ValueError("The number of subjects is too small.")

    test_data = _fisher_z_transform(rs)

    stat_results = _ttest_1samp_along_axis(
        data=test_data,
        axis=0,
        popmean=0,
        alternative=alternative,
        min_n=_MIN_SUBS,
    )

    return stat_results


def stats_fmri(corrs, alternative="greater"):
    """
    Conduct statistical analysis for fMRI searchlight correlation results.

    Parameters
    ----------
    corrs : array_like, shape [n_subs, n_x, n_y, n_z]
        Correlation coefficients.

        n_subs, n_x, n_y, and n_z represent the number of subjects and
        searchlight units along the x, y, and z axes.

    alternative : {"two-sided", "less", "greater"}, default="greater"
        Alternative hypothesis for one-sample t-tests against zero.

    Returns
    -------
    stat_results : ndarray, shape [n_x, n_y, n_z, 2]
        Statistical results.

        The last dimension contains:

        - stat_results[..., 0] : t-value
        - stat_results[..., 1] : p-value

    Notes
    -----
    n_subs must be at least 6.

    Fisher z transformation is always applied before the t-test.

    This function can be used for subject-level fMRI searchlight results,
    such as searchlight NPS or searchlight RDM correlations.
    """

    rs = _validate_r_values(
        corrs,
        expected_ndim=4,
        data_name="corrs",
    )

    if rs.shape[0] < _MIN_SUBS:
        raise ValueError("The number of subjects is too small.")

    test_data = _fisher_z_transform(rs)

    stat_results = _ttest_1samp_along_axis(
        data=test_data,
        axis=0,
        popmean=0,
        alternative=alternative,
        min_n=_MIN_SUBS,
    )

    return stat_results


def stats_iscfmri(corrs, alternative="greater"):
    """
    Conduct statistical analysis for fMRI ISC searchlight results.

    Parameters
    ----------
    corrs : array_like, shape [n_ts, n_pairs, n_x, n_y, n_z]
        ISC correlation coefficients.

        n_ts, n_pairs, n_x, n_y, and n_z represent the number of time points,
        subject pairs, and searchlight units along the x, y, and z axes.

    alternative : {"two-sided", "less", "greater"}, default="greater"
        Alternative hypothesis for one-sample t-tests against zero.

    Returns
    -------
    stat_results : ndarray, shape [n_ts, n_x, n_y, n_z, 2]
        Statistical results.

        The last dimension contains:

        - stat_results[..., 0] : t-value
        - stat_results[..., 1] : p-value

    Notes
    -----
    n_pairs must be at least 6.

    Fisher z transformation is always applied before the t-test.

    The second dimension of corrs is the subject-pair dimension.
    This function performs one-sample t-tests across subject pairs.
    """

    rs = _validate_r_values(
        corrs,
        expected_ndim=5,
        data_name="corrs",
    )

    if rs.shape[1] < _MIN_PAIRS:
        raise ValueError("The number of subject pairs is too small.")

    test_data = _fisher_z_transform(rs)

    # Axis 1 is the subject-pair axis:
    # [n_ts, n_pairs, n_x, n_y, n_z] -> [n_ts, n_x, n_y, n_z, 2]
    stat_results = _ttest_1samp_along_axis(
        data=test_data,
        axis=1,
        popmean=0,
        alternative=alternative,
        min_n=_MIN_PAIRS,
    )

    return stat_results