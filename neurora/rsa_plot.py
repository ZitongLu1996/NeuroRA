# -*- coding: utf-8 -*-

' a module for plotting the NeuroRA results '

__author__ = 'Zitong Lu'

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import ttest_1samp, ttest_rel
from nilearn import plotting, datasets, surface
import nibabel as nib
from neurora.stuff import get_affine, get_bg_ch2, get_bg_ch2bet, correct_by_threshold, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided, \
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, \
    smooth_1d_gaussian, smooth_2d_gaussian
from decimal import Decimal


def _validate_rdm_for_plot(rdm):
    """
    Validate an RDM for plotting.

    Parameters
    ----------
    rdm : array_like, shape [n_cons, n_cons]
        Representational dissimilarity matrix.

    Returns
    -------
    rdm : ndarray, shape [n_cons, n_cons]
        Validated RDM.
    """

    rdm = np.asarray(rdm, dtype=float)

    if rdm.ndim != 2:
        raise ValueError("rdm should be a 2-D array with shape [n_cons, n_cons].")

    if rdm.shape[0] != rdm.shape[1]:
        raise ValueError("rdm should be a square matrix with shape [n_cons, n_cons].")

    if rdm.shape[0] < 3:
        raise ValueError("rdm should contain at least 3 conditions for plotting.")

    return rdm


def _validate_lim(lim):
    """
    Validate color limit.

    Parameters
    ----------
    lim : tuple, list, or None
        Color limits.

    Returns
    -------
    lim : tuple or None
        Validated color limits.
    """

    if lim is None:
        return None

    if len(lim) != 2:
        raise ValueError("lim should have length 2: (vmin, vmax).")

    vmin, vmax = float(lim[0]), float(lim[1])

    if vmax <= vmin:
        raise ValueError("lim[1] should be larger than lim[0].")

    return vmin, vmax


def _validate_conditions(conditions, n_cons):
    """
    Validate condition labels.

    Parameters
    ----------
    conditions : list of str or None
        Condition labels.

    n_cons : int
        Number of conditions.

    Returns
    -------
    conditions : list of str or None
        Validated condition labels.
    """

    if conditions is None:
        return None

    if len(conditions) != n_cons:
        raise ValueError("conditions should have length n_cons.")

    return list(conditions)


def _percentile_transform_rdm(rdm):
    """
    Transform RDM values into percentile ranks.

    Parameters
    ----------
    rdm : ndarray, shape [n_cons, n_cons]
        Input RDM.

    Returns
    -------
    transformed_rdm : ndarray, shape [n_cons, n_cons]
        Percentile-transformed RDM.
    """

    rdm = np.asarray(rdm, dtype=float)
    n_cons = rdm.shape[0]

    values = rdm.ravel()
    valid_mask = np.isfinite(values)

    transformed = np.full_like(values, np.nan, dtype=float)

    valid_values = values[valid_mask]

    if valid_values.size == 0:
        return transformed.reshape(rdm.shape)

    order = np.argsort(valid_values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(valid_values.size, dtype=float)

    if valid_values.size > 1:
        percentile_values = ranks * 100.0 / (valid_values.size - 1)
    else:
        percentile_values = np.zeros_like(ranks, dtype=float)

    transformed[valid_mask] = percentile_values

    return transformed.reshape((n_cons, n_cons))


def _validate_tbyt_data(data, name="data", allow_last_dim_2=False):
    """
    Validate time-by-time subject-level data.

    Parameters
    ----------
    data : array_like
        Input data.

    name : str, default="data"
        Name used in error messages.

    allow_last_dim_2 : bool, default=False
        Whether to allow shape [n_subs, n_ts, 2].

    Returns
    -------
    data : ndarray, shape [n_subs, n_ts]
        Validated data.
    """

    data = np.asarray(data, dtype=float)

    if allow_last_dim_2 and data.ndim == 3 and data.shape[-1] == 2:
        data = data[:, :, 0]

    if data.ndim != 2:
        raise ValueError(f"{name} should have shape [n_subs, n_ts].")

    if data.shape[0] < 2:
        raise ValueError("At least two subjects are required.")

    return data


def _get_time_vector(n_ts, start_time=0, time_interval=0.01):
    """
    Create a time vector for time-by-time results.

    Parameters
    ----------
    n_ts : int
        Number of time points.

    start_time : float, default=0
        Time of the first data point.

    time_interval : float, default=0.01
        Time interval between adjacent data points.

    Returns
    -------
    times : ndarray, shape [n_ts]
        Time vector.
    """

    if not isinstance(time_interval, (int, float)) or time_interval <= 0:
        raise ValueError("time_interval should be a positive number.")

    return start_time + np.arange(n_ts) * float(time_interval)


def _get_stats_indices(times, stats_time):
    """
    Get indices corresponding to the statistical testing time window.

    Parameters
    ----------
    times : ndarray, shape [n_ts]
        Time vector.

    stats_time : tuple or list, length 2
        Statistical time window [start, end].

        The start is inclusive and the end is exclusive.

    Returns
    -------
    start_idx : int
        Start index.

    end_idx : int
        End index.
    """

    if len(stats_time) != 2:
        raise ValueError("stats_time should have length 2: (start, end).")

    stats_start = float(stats_time[0])
    stats_end = float(stats_time[1])

    if stats_end <= stats_start:
        raise ValueError("stats_time[1] should be larger than stats_time[0].")

    start_idx = int(np.searchsorted(times, stats_start, side="left"))
    end_idx = int(np.searchsorted(times, stats_end, side="left"))

    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, times.size)

    if end_idx <= start_idx:
        raise ValueError("stats_time does not overlap with the time vector.")

    return start_idx, end_idx


def _smooth_subject_timecourses(data, smooth=True, smooth_window=5):
    """
    Smooth subject-level time courses.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_ts]
        Subject-level time courses.

    smooth : bool, default=True
        Whether to smooth.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    Returns
    -------
    smoothed_data : ndarray, shape [n_subs, n_ts]
        Smoothed data.
    """

    if not isinstance(smooth, bool):
        raise TypeError("smooth should be True or False.")

    data = np.asarray(data, dtype=float)

    if not smooth:
        return np.array(data, copy=True)

    smoothed_data = np.zeros_like(data, dtype=float)

    for sub in range(data.shape[0]):
        smoothed_data[sub] = smooth_1d_gaussian(
            data[sub],
            smooth_window=smooth_window,
        )

    return smoothed_data


def _compute_tbyt_sem(data):
    """
    Compute mean and SEM across subjects.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_ts]
        Subject-level data.

    Returns
    -------
    avg : ndarray, shape [n_ts]
        Mean time course.

    sem : ndarray, shape [n_ts]
        SEM time course.
    """

    avg = np.nanmean(data, axis=0)

    n_valid = np.sum(np.isfinite(data), axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n_valid)

    sem[n_valid < 2] = np.nan

    return avg, sem


def _compute_tbyt_significance(data, level=0, p=0.05, cbpt=True, clusterp=0.05,
                               stats_time_idx=None, n_iter=1000):
    """
    Compute time-by-time significance.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_ts]
        Subject-level data.

    level : float, default=0
        Null hypothesis level.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_time_idx : tuple of int or None, default=None
        Statistical testing index range (start_idx, end_idx).

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    Returns
    -------
    ps : ndarray, shape [n_ts]
        Binary significance vector.

        1 means significant, 0 means not significant.
    """

    if not isinstance(cbpt, bool):
        raise TypeError("cbpt should be True or False.")

    if stats_time_idx is None:
        stats_time_idx = (0, data.shape[1])

    start_idx, end_idx = stats_time_idx

    ps = np.zeros(data.shape[1], dtype=float)

    if cbpt:
        ps_stats = clusterbased_permutation_1d_1samp_1sided(
            data[:, start_idx:end_idx],
            level=level,
            p_threshold=p,
            clusterp_threshold=clusterp,
            n_iter=n_iter,
        )

        ps[start_idx:end_idx] = ps_stats

    else:
        for t in range(start_idx, end_idx):
            _, p_value = ttest_1samp(
                data[:, t],
                level,
                nan_policy="omit",
                alternative="greater",
            )

            if np.isfinite(p_value) and p_value < p:
                ps[t] = 1

    return ps


def _print_significant_time_windows(ps, times, time_interval):
    """
    Print significant time windows.

    Parameters
    ----------
    ps : ndarray, shape [n_ts]
        Binary significance vector.

    times : ndarray, shape [n_ts]
        Time vector.

    time_interval : float
        Time interval between adjacent data points.
    """

    ps = np.asarray(ps)

    print("\nSignificant time-windows:")

    if not np.any(ps == 1):
        print("None")
        return

    in_cluster = False
    cluster_start = None

    for t in range(ps.size):
        if ps[t] == 1 and not in_cluster:
            in_cluster = True
            cluster_start = times[t]

        is_last = t == ps.size - 1
        next_is_zero = (not is_last) and ps[t + 1] != 1

        if ps[t] == 1 and (is_last or next_is_zero):
            cluster_end = times[t] + time_interval

            print(
                f"{int(round(cluster_start * 1000))}ms to "
                f"{int(round(cluster_end * 1000))}ms"
            )

            in_cluster = False


def _plot_tbyt_result(data, times, ps, level=0, color="r", xlim=None, ylim=None,
                      xlabel="Time (s)", ylabel="Value", figsize=(6.4, 3.6), x0=0, y0=None,
                      labelpad=0, ticksize=12, fontsize=16, markersize=2,
                      title=None, title_fontsize=16, avgshow=False, show=True):
    """
    Plot time-by-time result with SEM and significance markers.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_ts]
        Subject-level data.

    times : ndarray, shape [n_ts]
        Time vector.

    ps : ndarray, shape [n_ts]
        Binary significance vector.

    level : float, default=0
        Null hypothesis level.

    color : matplotlib color, default="r"
        Plot color.

    xlim : tuple or None
        X-axis limits.

    ylim : tuple or None
        Y-axis limits.

    xlabel, ylabel : str
        Axis labels.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    y0 : float or None, default=None
        Y position of x-axis spine. If None, uses level.

    labelpad : int or float, default=0
        Y-label padding.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    markersize : int or float, default=2
        Significant marker size.

    title : str or None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    avgshow : bool, default=False
        Whether to show the mean curve line.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes.Axes
        Axes object.
    """

    if y0 is None:
        y0 = level

    avg, sem = _compute_tbyt_sem(data)

    fig, ax = plt.subplots(figsize=figsize)

    if avgshow:
        ax.plot(times, avg, color=color, alpha=0.95)

    ax.fill_between(
        times,
        avg + sem,
        avg - sem,
        facecolor=color,
        alpha=0.35,
    )

    if ylim is not None:
        yminlim, ymaxlim = ylim
    else:
        yminlim, ymaxlim = ax.get_ylim()

    # Plot significant markers and light fill-to-baseline regions.
    for t, is_sig in enumerate(ps):
        if is_sig == 1:
            ax.plot(
                times[t],
                (ymaxlim - yminlim) * 0.95 + yminlim,
                "s",
                color=color,
                alpha=0.8,
                markersize=markersize,
            )

            x1 = times[t] - 0.5 * (times[1] - times[0]) if times.size > 1 else times[t]
            x2 = times[t] + 0.5 * (times[1] - times[0]) if times.size > 1 else times[t]

            ax.fill_between(
                [x1, x2],
                [avg[t] - sem[t], avg[t] - sem[t]],
                [level, level],
                facecolor=color,
                alpha=0.12,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))

    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", y0))

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labelsize=ticksize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _validate_tbyt_multiple_data(data, name="data", allow_last_dim_2=False):
    """
    Validate multiple-condition time-by-time data.

    Parameters
    ----------
    data : array_like
        Input data.

    name : str, default="data"
        Name used in error messages.

    allow_last_dim_2 : bool, default=False
        Whether to allow shape [n_conditions, n_subs, n_ts, 2].

    Returns
    -------
    data : ndarray, shape [n_conditions, n_subs, n_ts]
        Validated data.
    """

    data = np.asarray(data, dtype=float)

    if allow_last_dim_2 and data.ndim == 4 and data.shape[-1] == 2:
        data = data[:, :, :, 0]

    if data.ndim != 3:
        raise ValueError(
            f"{name} should have shape [n_conditions, n_subs, n_ts]."
        )

    if data.shape[0] < 1:
        raise ValueError("At least one condition is required.")

    if data.shape[1] < 2:
        raise ValueError("At least two subjects are required.")

    return data


def _validate_colors(colors, n_conditions):
    """
    Validate colors for multiple-condition plots.

    Parameters
    ----------
    colors : list or None
        Color list.

    n_conditions : int
        Number of conditions.

    Returns
    -------
    colors : list
        Validated color list.
    """

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_conditions)]
        return colors

    if len(colors) != n_conditions:
        raise ValueError("colors should have length n_conditions.")

    return list(colors)


def _validate_condition_labels(condition_labels, n_conditions):
    """
    Validate condition labels.

    Parameters
    ----------
    condition_labels : list of str or None
        Condition labels.

    n_conditions : int
        Number of conditions.

    Returns
    -------
    condition_labels : list of str
        Validated condition labels.
    """

    if condition_labels is None:
        return [f"Condition {i + 1}" for i in range(n_conditions)]

    if len(condition_labels) != n_conditions:
        raise ValueError("condition_labels should have length n_conditions.")

    return list(condition_labels)


def _plot_tbyt_multiple_result(data, times, ps, colors, condition_labels=None, level=0,
                               xlim=None, ylim=None, xlabel="Time (s)", ylabel="Value",
                               figsize=(6.4, 3.6), x0=0, y0=None, labelpad=0, ticksize=12,
                               fontsize=16, title=None, title_fontsize=16, sig_linewidth=3,
                               nonsig_linewidth=1, sem_alpha=0.2, nonsig_alpha=0.8, sig_alpha=1.0,
                               nonsig_linestyle="--", legend=True, show=True):
    """
    Plot multiple condition-level time courses with SEM and significance-coded lines.

    Parameters
    ----------
    data : ndarray, shape [n_conditions, n_subs, n_ts]
        Subject-level data.

    times : ndarray, shape [n_ts]
        Time vector.

    ps : ndarray, shape [n_conditions, n_ts]
        Binary significance vectors.

    colors : list
        Colors for conditions.

    condition_labels : list of str or None, default=None
        Condition labels.

    level : float, default=0
        Null hypothesis level.

    xlim : tuple or None
        X-axis limits.

    ylim : tuple or None
        Y-axis limits.

    xlabel, ylabel : str
        Axis labels.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    y0 : float or None, default=None
        Y position of x-axis spine. If None, uses level.

    sig_linewidth : int or float, default=3
        Line width for significant time points.

    nonsig_linewidth : int or float, default=1
        Line width for non-significant time points.

    sem_alpha : float, default=0.2
        Alpha for SEM shading.

    nonsig_alpha : float, default=0.8
        Alpha for non-significant dashed lines.

    sig_alpha : float, default=1.0
        Alpha for significant solid lines.

    nonsig_linestyle : str, default="--"
        Line style for non-significant average lines.

    legend : bool, default=True
        Whether to show legend.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes.Axes
        Axes object.
    """

    if y0 is None:
        y0 = level

    n_conditions = data.shape[0]

    if condition_labels is None:
        condition_labels = [f"Condition {i + 1}" for i in range(n_conditions)]

    fig, ax = plt.subplots(figsize=figsize)

    for cond in range(n_conditions):
        avg, sem = _compute_tbyt_sem(data[cond])
        color = colors[cond]
        label = condition_labels[cond]

        # SEM shading
        ax.fill_between(
            times,
            avg + sem,
            avg - sem,
            facecolor=color,
            alpha=sem_alpha,
            linewidth=0,
        )

        # Full non-significant-style average line.
        # Significant parts will be overlaid below.
        ax.plot(
            times,
            avg,
            color=color,
            linestyle=nonsig_linestyle,
            linewidth=nonsig_linewidth,
            alpha=nonsig_alpha,
            label=label,
        )

        # Overlay significant clusters as solid thick lines.
        sig_mask = ps[cond] == 1

        if np.any(sig_mask):
            in_cluster = False
            start_idx = None

            for t in range(times.size):
                if sig_mask[t] and not in_cluster:
                    in_cluster = True
                    start_idx = t

                is_last = t == times.size - 1
                next_is_zero = (not is_last) and (not sig_mask[t + 1])

                if sig_mask[t] and (is_last or next_is_zero):
                    end_idx = t

                    ax.plot(
                        times[start_idx:end_idx + 1],
                        avg[start_idx:end_idx + 1],
                        color=color,
                        linestyle="-",
                        linewidth=sig_linewidth,
                        alpha=sig_alpha,
                    )

                    in_cluster = False

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))

    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", y0))

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labelsize=ticksize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if legend:
        ax.legend(frameon=False, fontsize=ticksize)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _validate_ct_data(data, name="data"):
    """
    Validate cross-temporal subject-level data.

    Parameters
    ----------
    data : array_like, shape [n_subs, n_tsx, n_tsy]
        Cross-temporal data.

    name : str, default="data"
        Name used in error messages.

    Returns
    -------
    data : ndarray, shape [n_subs, n_tsx, n_tsy]
        Validated data.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 3:
        raise ValueError(f"{name} should have shape [n_subs, n_tsx, n_tsy].")

    if data.shape[0] < 2:
        raise ValueError("At least two subjects are required.")

    return data


def _smooth_subject_2d_maps(data, smooth=True, smooth_window=5):
    """
    Smooth subject-level 2-D maps.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_tsx, n_tsy]
        Subject-level 2-D maps.

    smooth : bool, default=True
        Whether to smooth.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    Returns
    -------
    smoothed_data : ndarray, shape [n_subs, n_tsx, n_tsy]
        Smoothed data.
    """

    if not isinstance(smooth, bool):
        raise TypeError("smooth should be True or False.")

    data = np.asarray(data, dtype=float)

    if not smooth:
        return np.array(data, copy=True)

    smoothed_data = np.zeros_like(data, dtype=float)

    for sub in range(data.shape[0]):
        smoothed_data[sub] = smooth_2d_gaussian(
            data[sub],
            smooth_window=smooth_window,
        )

    return smoothed_data


def _get_time_edges(times, time_interval):
    """
    Create bin edges for imshow/pcolormesh from time-point centers.

    Parameters
    ----------
    times : ndarray, shape [n_ts]
        Time-point centers.

    time_interval : float
        Time interval between adjacent time points.

    Returns
    -------
    edges : ndarray, shape [n_ts + 1]
        Time bin edges.
    """

    times = np.asarray(times, dtype=float)
    half_step = float(time_interval) / 2.0

    edges = np.concatenate(
        [
            times - half_step,
            [times[-1] + half_step],
        ]
    )

    return edges


def _compute_ct_significance(data, level=0.5, p=0.05, cbpt=True, clusterp=0.05,
                             stats_idx_x=None, stats_idx_y=None, n_iter=1000):
    """
    Compute significance for cross-temporal maps.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_tsx, n_tsy]
        Subject-level cross-temporal maps.

    level : float, default=0.5
        Null hypothesis level.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_idx_x : tuple of int or None, default=None
        Training-time statistical index range.

    stats_idx_y : tuple of int or None, default=None
        Test-time statistical index range.

    n_iter : int, default=1000
        Number of permutations.

    Returns
    -------
    ps : ndarray, shape [n_tsx, n_tsy]
        Binary significance map.
    """

    if not isinstance(cbpt, bool):
        raise TypeError("cbpt should be True or False.")

    n_subs, n_tsx, n_tsy = data.shape

    if stats_idx_x is None:
        stats_idx_x = (0, n_tsx)

    if stats_idx_y is None:
        stats_idx_y = (0, n_tsy)

    x_start, x_end = stats_idx_x
    y_start, y_end = stats_idx_y

    ps = np.zeros((n_tsx, n_tsy), dtype=float)

    if cbpt:
        ps_stats = clusterbased_permutation_2d_1samp_1sided(
            data[:, x_start:x_end, y_start:y_end],
            level=level,
            p_threshold=p,
            clusterp_threshold=clusterp,
            n_iter=n_iter,
        )

        ps[x_start:x_end, y_start:y_end] = ps_stats

    else:
        for ix in range(x_start, x_end):
            for iy in range(y_start, y_end):
                _, p_value = ttest_1samp(
                    data[:, ix, iy],
                    level,
                    nan_policy="omit",
                    alternative="greater",
                )

                if np.isfinite(p_value) and p_value < p:
                    ps[ix, iy] = 1

    return ps


def _validate_plot_img(img):
    """
    Validate and load a NIfTI image.

    Parameters
    ----------
    img : str or Niimg-like object
        Input image.

    Returns
    -------
    img_obj : Niimg-like object
        Loaded image object.

    data : ndarray
        Image data.
    """

    img_obj = nib.load(img) if isinstance(img, str) else img
    data = img_obj.get_fdata()

    if data.ndim != 3:
        raise ValueError("img should be a 3-D NIfTI image.")

    return img_obj, data


def _has_valid_brainrsa_results(data):
    """
    Check whether an RSA brain map contains valid non-zero results.

    Parameters
    ----------
    data : ndarray
        Brain map data.

    Returns
    -------
    has_results : bool
        Whether the map has valid finite non-zero values.
    """

    return bool(np.any(np.isfinite(data) & (data != 0)))


def _prepare_brainrsa_img(img, nvoxels_threshold=None, connectivity=1):
    """
    Load and optionally cluster-size threshold a brain RSA image.

    Parameters
    ----------
    img : str or Niimg-like object
        Input image.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

    connectivity : int, default=1
        Connectivity used for cluster labeling.

    Returns
    -------
    img_obj : Niimg-like object
        Prepared image object.

    data : ndarray
        Prepared image data.

    has_results : bool
        Whether the image contains valid results.
    """

    img_obj, data = _validate_plot_img(img)

    if not _has_valid_brainrsa_results(data):
        return img_obj, data, False

    if nvoxels_threshold is not None:
        if not isinstance(nvoxels_threshold, int) or nvoxels_threshold < 1:
            raise ValueError("nvoxels_threshold should be a positive integer or None.")

        affine = img_obj.affine
        data = correct_by_threshold(
            data,
            threshold=nvoxels_threshold,
            connectivity=connectivity,
        )
        img_obj = nib.Nifti1Image(data, affine)

    has_results = _has_valid_brainrsa_results(data)

    return img_obj, data, has_results


def _validate_clim(vmin, vmax):
    """
    Validate color limits.

    Parameters
    ----------
    vmin : float or None
        Lower color limit.

    vmax : float or None
        Upper color limit.

    Returns
    -------
    vmin, vmax : float or None
        Validated color limits.
    """

    if vmin is not None:
        vmin = float(vmin)

    if vmax is not None:
        vmax = float(vmax)

    if vmin is not None and vmax is not None and vmax <= vmin:
        raise ValueError("vmax should be larger than vmin.")

    return vmin, vmax


def _validate_cut_coords(cut_coords):
    """
    Validate cut coordinates for montage plotting.

    Parameters
    ----------
    cut_coords : list or tuple, length 3
        Cut coordinates for x, y, z displays.

    Returns
    -------
    cut_x, cut_y, cut_z
        Validated cut coordinates.
    """

    if len(cut_coords) != 3:
        raise ValueError("cut_coords should have length 3: [x, y, z].")

    return cut_coords[0], cut_coords[1], cut_coords[2]


def _maybe_show(show=True):
    """
    Show current matplotlib figures if requested.
    """

    if not isinstance(show, bool):
        raise TypeError("show should be True or False.")

    if show:
        plt.show()


def plot_rdm(rdm, percentile=False, lim=(0, 1), conditions=None, con_fontsize=12, cmap=None,
             title=None, title_fontsize=16, colorbar_label=None, colorbar_fontsize=16,
             show_values=False, value_fontsize=10, value_format=".4f", value_color="blue",
             figsize=None, show=True):
    """
    Plot a representational dissimilarity matrix (RDM).

    Parameters
    ----------
    rdm : array_like, shape [n_cons, n_cons]
        Representational dissimilarity matrix.

    percentile : bool, default=False
        Whether to display RDM values as percentile ranks.

    lim : tuple of float or None, default=(0, 1)
        Color limits.

        If None, matplotlib determines the limits automatically.

    conditions : list of str or None, default=None
        Condition labels.

    con_fontsize : int or float, default=12
        Font size of condition labels.

    cmap : matplotlib colormap or str or None, default=None
        Colormap.

        If None, "jet" is used.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Font size of the title.

    colorbar_label : str or None, default=None
        Colorbar label.

        If None, a default label is used.

    colorbar_fontsize : int or float, default=16
        Font size of colorbar tick labels and label.

    show_values : bool, default=False
        Whether to show numerical values inside RDM cells.

    value_fontsize : int or float, default=10
        Font size of displayed cell values.

    value_format : str, default=".4f"
        Format string for displayed values.

    value_color : str, default="blue"
        Color of displayed values.

    figsize : tuple or None, default=None
        Figure size.

    show : bool, default=True
        Whether to show the figure immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes.Axes
        Axes object.
    """

    rdm = _validate_rdm_for_plot(rdm)
    n_cons = rdm.shape[0]

    if not isinstance(percentile, bool):
        raise TypeError("percentile should be True or False.")

    if not isinstance(show_values, bool):
        raise TypeError("show_values should be True or False.")

    if not isinstance(show, bool):
        raise TypeError("show should be True or False.")

    conditions = _validate_conditions(conditions, n_cons)

    plot_rdm_data = np.array(rdm, copy=True)

    if percentile:
        plot_rdm_data = _percentile_transform_rdm(plot_rdm_data)
        clim = (0, 100)
        default_colorbar_label = "Dissimilarity (percentile)"
    else:
        clim = _validate_lim(lim)
        default_colorbar_label = "Dissimilarity"

    if colorbar_label is None:
        colorbar_label = default_colorbar_label

    if cmap is None:
        cmap = "jet"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        plot_rdm_data,
        cmap=cmap,
        vmin=None if clim is None else clim[0],
        vmax=None if clim is None else clim[1],
        origin="upper",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    if conditions is not None:
        ticks = np.arange(n_cons)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels(
            conditions,
            fontsize=con_fontsize,
            rotation=30,
            ha="right",
        )

        ax.set_yticklabels(
            conditions,
            fontsize=con_fontsize,
        )

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_values:
        for i in range(n_cons):
            for j in range(n_cons):
                value = plot_rdm_data[i, j]

                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        color=value_color,
                        fontsize=value_fontsize,
                    )

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_tbyt_similarity(similarities, start_time=0, time_interval=0.01, smooth=True, smooth_window=5,
                         p=0.05, cbpt=True, clusterp=0.05, stats_time=None, color="r",
                         xlim=None, ylim=(-0.1, 0.8), xlabel="Time (s)", ylabel="Representational Similarity",
                         figsize=(6.4, 3.6), x0=0, labelpad=0, ticksize=12, fontsize=16, markersize=2,
                         title=None, title_fontsize=16, avgshow=False, n_iter=1000, show=True):
    """
    Plot time-by-time representational similarities with statistical results.

    Parameters
    ----------
    similarities : array_like, shape [n_subs, n_ts] or [n_subs, n_ts, 2]
        Subject-level time-by-time similarities.

        If shape is [n_subs, n_ts, 2], only similarities[..., 0] is used.

    start_time : float, default=0
        Time of the first data point.

    time_interval : float, default=0.01
        Time interval between adjacent data points.

    smooth : bool, default=True
        Whether to smooth each subject's time course.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_time : tuple or list of float or None, default=None
        Time window for statistical analysis.

        If None, the full time range is used.

    color : matplotlib color, default="r"
        Plot color.

    xlim : tuple or None, default=None
        X-axis limits. If None, uses the full time range.

    ylim : tuple, default=(-0.1, 0.8)
        Y-axis limits.

    xlabel : str, default="Time (s)"
        X-axis label.

    ylabel : str, default="Representational Similarity"
        Y-axis label.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    labelpad : int or float, default=0
        Y-label padding.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    markersize : int or float, default=2
        Significant marker size.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    avgshow : bool, default=False
        Whether to show the mean curve line.

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    ps : ndarray, shape [n_ts]
        Binary significance vector.
    """

    similarities = _validate_tbyt_data(
        similarities,
        name="similarities",
        allow_last_dim_2=True,
    )

    nsubs, n_ts = similarities.shape

    times = _get_time_vector(
        n_ts=n_ts,
        start_time=start_time,
        time_interval=time_interval,
    )

    if stats_time is None:
        stats_time = (times[0], times[-1] + time_interval)

    stats_idx = _get_stats_indices(
        times=times,
        stats_time=stats_time,
    )

    data = _smooth_subject_timecourses(
        similarities,
        smooth=smooth,
        smooth_window=smooth_window,
    )

    ps = _compute_tbyt_significance(
        data=data,
        level=0,
        p=p,
        cbpt=cbpt,
        clusterp=clusterp,
        stats_time_idx=stats_idx,
        n_iter=n_iter,
    )

    _print_significant_time_windows(
        ps=ps,
        times=times,
        time_interval=time_interval,
    )

    if xlim is None:
        xlim = (times[0], times[-1])

    _plot_tbyt_result(
        data=data,
        times=times,
        ps=ps,
        level=0,
        color=color,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        x0=x0,
        y0=0,
        labelpad=labelpad,
        ticksize=ticksize,
        fontsize=fontsize,
        markersize=markersize,
        title=title,
        title_fontsize=title_fontsize,
        avgshow=avgshow,
        show=show,
    )

    return ps


def plot_tbyt_decoding_acc(acc, start_time=0, time_interval=0.01, smooth=True, smooth_window=5,
                           chance=0.5, p=0.05, cbpt=True, clusterp=0.05, stats_time=None, color="r",
                           xlim=None, ylim=(0.4, 0.8), xlabel="Time (s)", ylabel="Decoding Accuracy",
                           figsize=(6.4, 3.6), x0=0, labelpad=0, ticksize=12, fontsize=16, markersize=2,
                           title=None, title_fontsize=16, avgshow=False, n_iter=1000, show=True):
    """
    Plot time-by-time decoding accuracies with statistical results.

    Parameters
    ----------
    acc : array_like, shape [n_subs, n_ts]
        Subject-level time-by-time decoding accuracies.

    start_time : float, default=0
        Time of the first data point.

    time_interval : float, default=0.01
        Time interval between adjacent data points.

    smooth : bool, default=True
        Whether to smooth each subject's time course.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    chance : float, default=0.5
        Chance level.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_time : tuple or list of float or None, default=None
        Time window for statistical analysis.

        If None, the full time range is used.

    color : matplotlib color, default="r"
        Plot color.

    xlim : tuple or None, default=None
        X-axis limits. If None, uses the full time range.

    ylim : tuple, default=(0.4, 0.8)
        Y-axis limits.

    xlabel : str, default="Time (s)"
        X-axis label.

    ylabel : str, default="Decoding Accuracy"
        Y-axis label.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    labelpad : int or float, default=0
        Y-label padding.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    markersize : int or float, default=2
        Significant marker size.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    avgshow : bool, default=False
        Whether to show the mean curve line.

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    ps : ndarray, shape [n_ts]
        Binary significance vector.
    """

    acc = _validate_tbyt_data(
        acc,
        name="acc",
        allow_last_dim_2=False,
    )

    nsubs, n_ts = acc.shape

    times = _get_time_vector(
        n_ts=n_ts,
        start_time=start_time,
        time_interval=time_interval,
    )

    if stats_time is None:
        stats_time = (times[0], times[-1] + time_interval)

    stats_idx = _get_stats_indices(
        times=times,
        stats_time=stats_time,
    )

    data = _smooth_subject_timecourses(
        acc,
        smooth=smooth,
        smooth_window=smooth_window,
    )

    ps = _compute_tbyt_significance(
        data=data,
        level=chance,
        p=p,
        cbpt=cbpt,
        clusterp=clusterp,
        stats_time_idx=stats_idx,
        n_iter=n_iter,
    )

    _print_significant_time_windows(
        ps=ps,
        times=times,
        time_interval=time_interval,
    )

    if xlim is None:
        xlim = (times[0], times[-1])

    _plot_tbyt_result(
        data=data,
        times=times,
        ps=ps,
        level=chance,
        color=color,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        x0=x0,
        y0=chance,
        labelpad=labelpad,
        ticksize=ticksize,
        fontsize=fontsize,
        markersize=markersize,
        title=title,
        title_fontsize=title_fontsize,
        avgshow=avgshow,
        show=show,
    )

    return ps


def plot_tbyt_similarity_multiple(similarities, start_time=0, time_interval=0.01, smooth=True, smooth_window=5,
                                  p=0.05, cbpt=True, clusterp=0.05, stats_time=None, colors=None,
                                  condition_labels=None, xlim=None, ylim=(-0.1, 0.8), xlabel="Time (s)",
                                  ylabel="Representational Similarity", figsize=(6.4, 3.6), x0=0, labelpad=0,
                                  ticksize=12, fontsize=16, title=None, title_fontsize=16, sig_linewidth=3,
                                  nonsig_linewidth=1, sem_alpha=0.2, nonsig_alpha=0.8, sig_alpha=1.0,
                                  nonsig_linestyle="--", legend=True, n_iter=1000, show=True):
    """
    Plot multiple time-by-time representational similarity results.

    Parameters
    ----------
    similarities : array_like, shape [n_conditions, n_subs, n_ts]
        Multiple-condition subject-level similarity time courses.

        Shape should be [n_conditions, n_subs, n_timepoints].

    start_time : float, default=0
        Time of the first data point.

    time_interval : float, default=0.01
        Time interval between adjacent data points.

    smooth : bool, default=True
        Whether to smooth each subject's time course.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_time : tuple or list of float or None, default=None
        Time window for statistical analysis.

        If None, the full time range is used.

    colors : list or None, default=None
        Colors for conditions.

        If None, matplotlib tab10 colors are used.

    condition_labels : list of str or None, default=None
        Condition labels for legend.

    xlim : tuple or None, default=None
        X-axis limits. If None, uses the full time range.

    ylim : tuple, default=(-0.1, 0.8)
        Y-axis limits.

    xlabel : str, default="Time (s)"
        X-axis label.

    ylabel : str, default="Representational Similarity"
        Y-axis label.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    labelpad : int or float, default=0
        Y-label padding.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    sig_linewidth : int or float, default=3
        Line width for significant time points.

    nonsig_linewidth : int or float, default=1
        Line width for non-significant time points.

    sem_alpha : float, default=0.2
        Alpha for SEM shading.

    nonsig_alpha : float, default=0.8
        Alpha for non-significant dashed lines.

    sig_alpha : float, default=1.0
        Alpha for significant solid lines.

    nonsig_linestyle : str, default="--"
        Line style for non-significant time points.

    legend : bool, default=True
        Whether to show legend.

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    ps : ndarray, shape [n_conditions, n_ts]
        Binary significance vectors.
    """

    similarities = _validate_tbyt_multiple_data(
        similarities,
        name="similarities",
        allow_last_dim_2=True,
    )

    n_conditions, n_subs, n_ts = similarities.shape

    colors = _validate_colors(colors, n_conditions)
    condition_labels = _validate_condition_labels(condition_labels, n_conditions)

    times = _get_time_vector(
        n_ts=n_ts,
        start_time=start_time,
        time_interval=time_interval,
    )

    if stats_time is None:
        stats_time = (times[0], times[-1] + time_interval)

    stats_idx = _get_stats_indices(
        times=times,
        stats_time=stats_time,
    )

    data = np.zeros_like(similarities, dtype=float)

    for cond in range(n_conditions):
        data[cond] = _smooth_subject_timecourses(
            similarities[cond],
            smooth=smooth,
            smooth_window=smooth_window,
        )

    ps = np.zeros((n_conditions, n_ts), dtype=float)

    for cond in range(n_conditions):
        ps[cond] = _compute_tbyt_significance(
            data=data[cond],
            level=0,
            p=p,
            cbpt=cbpt,
            clusterp=clusterp,
            stats_time_idx=stats_idx,
            n_iter=n_iter,
        )

        print(f"\n{condition_labels[cond]}:")
        _print_significant_time_windows(
            ps=ps[cond],
            times=times,
            time_interval=time_interval,
        )

    if xlim is None:
        xlim = (times[0], times[-1])

    _plot_tbyt_multiple_result(
        data=data,
        times=times,
        ps=ps,
        colors=colors,
        condition_labels=condition_labels,
        level=0,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        x0=x0,
        y0=0,
        labelpad=labelpad,
        ticksize=ticksize,
        fontsize=fontsize,
        title=title,
        title_fontsize=title_fontsize,
        sig_linewidth=sig_linewidth,
        nonsig_linewidth=nonsig_linewidth,
        sem_alpha=sem_alpha,
        nonsig_alpha=nonsig_alpha,
        sig_alpha=sig_alpha,
        nonsig_linestyle=nonsig_linestyle,
        legend=legend,
        show=show,
    )

    return ps


def plot_tbyt_decoding_acc_multiple(acc, start_time=0, time_interval=0.01, smooth=True, smooth_window=5,
                                    chance=0.5, p=0.05, cbpt=True, clusterp=0.05, stats_time=None, colors=None,
                                    condition_labels=None, xlim=None, ylim=(0.4, 0.8), xlabel="Time (s)",
                                    ylabel="Decoding Accuracy", figsize=(6.4, 3.6), x0=0, labelpad=0,
                                    ticksize=12, fontsize=16, title=None, title_fontsize=16, sig_linewidth=3,
                                    nonsig_linewidth=1, sem_alpha=0.2, nonsig_alpha=0.8, sig_alpha=1.0,
                                    nonsig_linestyle="--", legend=True, n_iter=1000, show=True):
    """
    Plot multiple time-by-time decoding accuracy results.

    Parameters
    ----------
    acc : array_like, shape [n_conditions, n_subs, n_ts]
        Multiple-condition subject-level decoding accuracy time courses.

        Shape should be [n_conditions, n_subs, n_timepoints].

    start_time : float, default=0
        Time of the first data point.

    time_interval : float, default=0.01
        Time interval between adjacent data points.

    smooth : bool, default=True
        Whether to smooth each subject's time course.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    chance : float, default=0.5
        Chance level.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_time : tuple or list of float or None, default=None
        Time window for statistical analysis.

        If None, the full time range is used.

    colors : list or None, default=None
        Colors for conditions.

        If None, matplotlib tab10 colors are used.

    condition_labels : list of str or None, default=None
        Condition labels for legend.

    xlim : tuple or None, default=None
        X-axis limits. If None, uses the full time range.

    ylim : tuple, default=(0.4, 0.8)
        Y-axis limits.

    xlabel : str, default="Time (s)"
        X-axis label.

    ylabel : str, default="Decoding Accuracy"
        Y-axis label.

    figsize : tuple, default=(6.4, 3.6)
        Figure size.

    x0 : float, default=0
        X position of y-axis spine.

    labelpad : int or float, default=0
        Y-label padding.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    sig_linewidth : int or float, default=3
        Line width for significant time points.

    nonsig_linewidth : int or float, default=1
        Line width for non-significant time points.

    sem_alpha : float, default=0.2
        Alpha for SEM shading.

    nonsig_alpha : float, default=0.8
        Alpha for non-significant dashed lines.

    sig_alpha : float, default=1.0
        Alpha for significant solid lines.

    nonsig_linestyle : str, default="--"
        Line style for non-significant time points.

    legend : bool, default=True
        Whether to show legend.

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    ps : ndarray, shape [n_conditions, n_ts]
        Binary significance vectors.
    """

    acc = _validate_tbyt_multiple_data(
        acc,
        name="acc",
        allow_last_dim_2=False,
    )

    n_conditions, n_subs, n_ts = acc.shape

    colors = _validate_colors(colors, n_conditions)
    condition_labels = _validate_condition_labels(condition_labels, n_conditions)

    times = _get_time_vector(
        n_ts=n_ts,
        start_time=start_time,
        time_interval=time_interval,
    )

    if stats_time is None:
        stats_time = (times[0], times[-1] + time_interval)

    stats_idx = _get_stats_indices(
        times=times,
        stats_time=stats_time,
    )

    data = np.zeros_like(acc, dtype=float)

    for cond in range(n_conditions):
        data[cond] = _smooth_subject_timecourses(
            acc[cond],
            smooth=smooth,
            smooth_window=smooth_window,
        )

    ps = np.zeros((n_conditions, n_ts), dtype=float)

    for cond in range(n_conditions):
        ps[cond] = _compute_tbyt_significance(
            data=data[cond],
            level=chance,
            p=p,
            cbpt=cbpt,
            clusterp=clusterp,
            stats_time_idx=stats_idx,
            n_iter=n_iter,
        )

        print(f"\n{condition_labels[cond]}:")
        _print_significant_time_windows(
            ps=ps[cond],
            times=times,
            time_interval=time_interval,
        )

    if xlim is None:
        xlim = (times[0], times[-1])

    _plot_tbyt_multiple_result(
        data=data,
        times=times,
        ps=ps,
        colors=colors,
        condition_labels=condition_labels,
        level=chance,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        x0=x0,
        y0=chance,
        labelpad=labelpad,
        ticksize=ticksize,
        fontsize=fontsize,
        title=title,
        title_fontsize=title_fontsize,
        sig_linewidth=sig_linewidth,
        nonsig_linewidth=nonsig_linewidth,
        sem_alpha=sem_alpha,
        nonsig_alpha=nonsig_alpha,
        sig_alpha=sig_alpha,
        nonsig_linestyle=nonsig_linestyle,
        legend=legend,
        show=show,
    )

    return ps


def plot_ct_decoding_acc(acc, start_timex=0, start_timey=0, time_intervalx=0.01, time_intervaly=0.01,
                         smooth=True, smooth_window=5, chance=0.5, p=0.05, cbpt=True, clusterp=0.05,
                         stats_timex=None, stats_timey=None, xlim=None, ylim=None, clim=(0.4, 0.8),
                         xlabel="Training Time (s)", ylabel="Test Time (s)", clabel="Decoding Accuracy",
                         figsize=(6.4, 4.8), cmap="viridis", contour_color="silver", contour_linewidth=3,
                         contour_linestyle="dashed", ticksize=12, fontsize=16, title=None, title_fontsize=16,
                         n_iter=1000, show=True):
    """
    Plot cross-temporal decoding accuracies.

    Parameters
    ----------
    acc : array_like, shape [n_subs, n_tsx, n_tsy]
        Subject-level cross-temporal decoding accuracies.

        n_tsx is the number of training time points.
        n_tsy is the number of test time points.

    start_timex : float, default=0
        Time of the first training time point.

    start_timey : float, default=0
        Time of the first test time point.

    time_intervalx : float, default=0.01
        Training-time interval between adjacent time points.

    time_intervaly : float, default=0.01
        Test-time interval between adjacent time points.

    smooth : bool, default=True
        Whether to smooth each subject's cross-temporal matrix.

    smooth_window : int, default=5
        Approximate Gaussian smoothing window in time points.

    chance : float, default=0.5
        Chance level.

    p : float, default=0.05
        Point-wise or cluster-forming p-value threshold.

    cbpt : bool, default=True
        Whether to use cluster-based permutation test.

    clusterp : float, default=0.05
        Cluster-level p-value threshold.

    stats_timex : tuple or list of float or None, default=None
        Training-time window for statistical analysis.

        If None, the full training-time range is used.

    stats_timey : tuple or list of float or None, default=None
        Test-time window for statistical analysis.

        If None, the full test-time range is used.

    xlim : tuple or None, default=None
        Training-time axis limits.

    ylim : tuple or None, default=None
        Test-time axis limits.

    clim : tuple, default=(0.4, 0.8)
        Color limits.

    xlabel : str, default="Training Time (s)"
        X-axis label.

    ylabel : str, default="Test Time (s)"
        Y-axis label.

    clabel : str, default="Decoding Accuracy"
        Colorbar label.

    figsize : tuple, default=(6.4, 4.8)
        Figure size.

    cmap : str or matplotlib colormap, default="viridis"
        Colormap.

    contour_color : matplotlib color, default="silver"
        Color of significance contour.

    contour_linewidth : int or float, default=3
        Line width of significance contour.

    contour_linestyle : str, default="dashed"
        Line style of significance contour.

    ticksize : int or float, default=12
        Tick label size.

    fontsize : int or float, default=16
        Axis label size.

    title : str or None, default=None
        Figure title.

    title_fontsize : int or float, default=16
        Title font size.

    n_iter : int, default=1000
        Number of permutations for cluster-based permutation test.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    ps : ndarray, shape [n_tsx, n_tsy]
        Binary significance map.
    """

    acc = _validate_ct_data(acc, name="acc")

    n_subs, n_tsx, n_tsy = acc.shape

    times_x = _get_time_vector(
        n_ts=n_tsx,
        start_time=start_timex,
        time_interval=time_intervalx,
    )

    times_y = _get_time_vector(
        n_ts=n_tsy,
        start_time=start_timey,
        time_interval=time_intervaly,
    )

    if stats_timex is None:
        stats_timex = (times_x[0], times_x[-1] + time_intervalx)

    if stats_timey is None:
        stats_timey = (times_y[0], times_y[-1] + time_intervaly)

    stats_idx_x = _get_stats_indices(
        times=times_x,
        stats_time=stats_timex,
    )

    stats_idx_y = _get_stats_indices(
        times=times_y,
        stats_time=stats_timey,
    )

    data = _smooth_subject_2d_maps(
        acc,
        smooth=smooth,
        smooth_window=smooth_window,
    )

    ps = _compute_ct_significance(
        data=data,
        level=chance,
        p=p,
        cbpt=cbpt,
        clusterp=clusterp,
        stats_idx_x=stats_idx_x,
        stats_idx_y=stats_idx_y,
        n_iter=n_iter,
    )

    avg = np.nanmean(data, axis=0)

    x_edges = _get_time_edges(
        times=times_x,
        time_interval=time_intervalx,
    )

    y_edges = _get_time_edges(
        times=times_y,
        time_interval=time_intervaly,
    )

    if xlim is None:
        xlim = (x_edges[0], x_edges[-1])

    if ylim is None:
        ylim = (y_edges[0], y_edges[-1])

    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        avg.T,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="auto",
    )

    # Significance contour.
    # ps is [train_time, test_time], so transpose it for plotting.
    if np.any(ps == 1):
        ax.contour(
            times_x,
            times_y,
            ps.T,
            levels=[0.5],
            colors=contour_color,
            linewidths=contour_linewidth,
            linestyles=contour_linestyle,
        )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label(clabel, fontsize=ticksize + 2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(labelsize=ticksize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    fig.tight_layout()

    if show:
        plt.show()

    return ps


def plot_brainrsa_regions(img, threshold=0, nvoxels_threshold=None, background=get_bg_ch2(),
                          vmin=None, vmax=None, cmap="hot", title="RSA Results", colorbar_label=None,
                          resampling_interpolation="continuous", connectivity=1, show=True):
    """
    Plot RSA-result regions using orthogonal brain cuts.

    Parameters
    ----------
    img : str or Niimg-like object
        NIfTI image of RSA results.

    threshold : float or None, default=0
        Display threshold passed to nilearn plotting.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

        If None, no cluster-size thresholding is applied.

    background : Niimg-like object or str, default=get_bg_ch2()
        Background image.

    vmin, vmax : float or None, default=None
        Color limits.

    cmap : str or matplotlib colormap, default="hot"
        Colormap.

    title : str or None, default="RSA Results"
        Plot title.

    colorbar_label : str or None, default=None
        Colorbar label.

    resampling_interpolation : str, default="continuous"
        Resampling interpolation passed to nilearn.

    connectivity : int, default=1
        Connectivity used for cluster-size thresholding.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    display
        Nilearn display object.
    """

    vmin, vmax = _validate_clim(vmin, vmax)

    img_obj, data, has_results = _prepare_brainrsa_img(
        img,
        nvoxels_threshold=nvoxels_threshold,
        connectivity=connectivity,
    )

    if not has_results:
        print("No valid results.")
        return None

    display = plotting.plot_roi(
        roi_img=img_obj,
        bg_img=background,
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        title=title,
        colorbar=True,
        resampling_interpolation=resampling_interpolation,
    )

    if colorbar_label is not None:
        display._cbar.set_label(colorbar_label)

    _maybe_show(show)

    return display


def plot_brainrsa_montage(img, threshold=0, nvoxels_threshold=None, cut_coords=(6, 6, 6),
                          background=get_bg_ch2bet(), vmin=None, vmax=None, cmap="hot",
                          title="RSA Results", colorbar_label=None, draw_cross=True, connectivity=1,
                          show=True):
    """
    Plot RSA results across sagittal, coronal, and axial cuts.

    Parameters
    ----------
    img : str or Niimg-like object
        NIfTI image of RSA results.

    threshold : float or None, default=0
        Display threshold passed to nilearn plotting.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

    cut_coords : list or tuple, length 3, default=(6, 6, 6)
        Cut coordinates for x, y, and z displays.

        Each element can be an int, float, list, tuple, or None.
        If an element is None or 0, that display direction is skipped.

    background : Niimg-like object or str, default=get_bg_ch2bet()
        Background image.

    vmin, vmax : float or None, default=None
        Color limits.

    cmap : str or matplotlib colormap, default="hot"
        Colormap.

    title : str or None, default="RSA Results"
        Base plot title.

    colorbar_label : str or None, default=None
        Colorbar label.

    draw_cross : bool, default=True
        Whether to draw crosshair.

    connectivity : int, default=1
        Connectivity used for cluster-size thresholding.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    displays : dict
        Nilearn display objects for x, y, and z cuts.
    """

    vmin, vmax = _validate_clim(vmin, vmax)
    cut_x, cut_y, cut_z = _validate_cut_coords(cut_coords)

    img_obj, data, has_results = _prepare_brainrsa_img(
        img,
        nvoxels_threshold=nvoxels_threshold,
        connectivity=connectivity,
    )

    if not has_results:
        print("No valid results.")
        return None

    displays = {}

    cut_specs = {
        "x": (cut_x, "sagittal"),
        "y": (cut_y, "coronal"),
        "z": (cut_z, "axial"),
    }

    for display_mode, (coords, name) in cut_specs.items():
        if coords is None or coords == 0:
            continue

        this_title = None if title is None else f"{title} - {name}"

        display = plotting.plot_stat_map(
            stat_map_img=img_obj,
            bg_img=background,
            display_mode=display_mode,
            cut_coords=coords,
            threshold=threshold,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=this_title,
            draw_cross=draw_cross,
            colorbar=True,
        )

        if colorbar_label is not None:
            display._cbar.set_label(colorbar_label)

        displays[display_mode] = display

    _maybe_show(show)

    return displays


def plot_brainrsa_glass(img, threshold=0, nvoxels_threshold=None, vmin=None, vmax=None,
                        cmap="hot", title="RSA Results", colorbar_label=None, black_bg=True,
                        draw_cross=True, connectivity=1, show=True):
    """
    Plot a glass-brain projection of RSA results.

    Parameters
    ----------
    img : str or Niimg-like object
        NIfTI image of RSA results.

    threshold : float or None, default=0
        Display threshold passed to nilearn plotting.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

    vmin, vmax : float or None, default=None
        Color limits.

    cmap : str or matplotlib colormap, default="hot"
        Colormap.

    title : str or None, default="RSA Results"
        Plot title.

    colorbar_label : str or None, default=None
        Colorbar label.

    black_bg : bool, default=True
        Whether to use a black background.

    draw_cross : bool, default=True
        Whether to draw crosshair.

    connectivity : int, default=1
        Connectivity used for cluster-size thresholding.

    show : bool, default=True
        Whether to show the plot immediately.

    Returns
    -------
    display
        Nilearn display object.
    """

    vmin, vmax = _validate_clim(vmin, vmax)

    img_obj, data, has_results = _prepare_brainrsa_img(
        img,
        nvoxels_threshold=nvoxels_threshold,
        connectivity=connectivity,
    )

    if not has_results:
        print("No valid results.")
        return None

    display = plotting.plot_glass_brain(
        img_obj,
        threshold=threshold,
        colorbar=True,
        title=title,
        black_bg=black_bg,
        draw_cross=draw_cross,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    if colorbar_label is not None:
        display._cbar.set_label(colorbar_label)

    _maybe_show(show)

    return display


def plot_brainrsa_surface(img, threshold=0.01, nvoxels_threshold=None, vmin=None, vmax=None,
                          cmap="hot", title="RSA Results", colorbar_label=None, mesh="fsaverage5",
                          interpolation="linear", views=("lateral", "lateral", "ventral", "dorsal"),
                          hemis=("left", "right", "both", "both"), darkness=0.7, connectivity=1,
                          show=True):
    """
    Plot RSA results on fsaverage cortical surface.

    Parameters
    ----------
    img : str or Niimg-like object
        NIfTI image of RSA results.

    threshold : float or None, default=0.01
        Display threshold for surface plotting.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

    vmin, vmax : float or None, default=None
        Color limits.

    cmap : str or matplotlib colormap, default="hot"
        Colormap.

    title : str or None, default="RSA Results"
        Base plot title.

    colorbar_label : str or None, default=None
        Colorbar label.

    mesh : str, default="fsaverage5"
        Fsaverage mesh density passed to nilearn.

    interpolation : str, default="linear"
        Interpolation method for vol_to_surf.

    views : tuple of str, default=("lateral", "lateral", "ventral", "dorsal")
        Surface views.

    hemis : tuple of str, default=("left", "right", "both", "both")
        Hemispheres to plot.

    darkness : float, default=0.7
        Background darkness.

    connectivity : int, default=1
        Connectivity used for cluster-size thresholding.

    show : bool, default=True
        Whether to show the plots immediately.

    Returns
    -------
    displays : list
        Nilearn surface display objects.
    """

    vmin, vmax = _validate_clim(vmin, vmax)

    img_obj, data, has_results = _prepare_brainrsa_img(
        img,
        nvoxels_threshold=nvoxels_threshold,
        connectivity=connectivity,
    )

    if not has_results:
        print("No valid results.")
        return None

    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    fsaverage_meshes = datasets.load_fsaverage()

    texture_l = surface.vol_to_surf(
        img_obj,
        fsaverage["pial_left"],
        interpolation=interpolation,
    )

    texture_r = surface.vol_to_surf(
        img_obj,
        fsaverage["pial_right"],
        interpolation=interpolation,
    )

    curv_sign = datasets.load_fsaverage_data(data_type="curvature")

    for hemi, hemi_data in curv_sign.data.parts.items():
        curv_sign.data.parts[hemi] = np.sign(hemi_data)

    surface_image = surface.SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img_obj,
    )

    displays = []

    plot_specs = [
        {
            "surf_mesh": fsaverage["infl_left"],
            "surf_map": texture_l,
            "hemi": "left",
            "bg_map": curv_sign.data.parts["left"],
            "view": views[0],
            "title": None if title is None else f"{title} - left",
        },
        {
            "surf_mesh": fsaverage["infl_right"],
            "surf_map": texture_r,
            "hemi": "right",
            "bg_map": curv_sign.data.parts["right"],
            "view": views[1],
            "title": None if title is None else f"{title} - right",
        },
        {
            "surf_mesh": fsaverage_meshes["inflated"],
            "surf_map": surface_image,
            "hemi": "both",
            "bg_map": curv_sign,
            "view": views[2],
            "title": None if title is None else f"{title} - ventral",
        },
        {
            "surf_mesh": fsaverage_meshes["inflated"],
            "surf_map": surface_image,
            "hemi": "both",
            "bg_map": curv_sign,
            "view": views[3],
            "title": None if title is None else f"{title} - dorsal",
        },
    ]

    for spec in plot_specs:
        display = plotting.plot_surf(
            surf_mesh=spec["surf_mesh"],
            surf_map=spec["surf_map"],
            hemi=spec["hemi"],
            bg_map=spec["bg_map"],
            view=spec["view"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            colorbar=True,
            darkness=darkness,
            title=spec["title"],
        )

        # plot_surf returns a figure-like display depending on nilearn version.
        # Colorbar label support is not always exposed consistently.
        if colorbar_label is not None:
            try:
                display._cbar.set_label(colorbar_label)
            except AttributeError:
                pass

        displays.append(display)

    _maybe_show(show)

    return displays


def plot_brainrsa_rlts(img, threshold=0.01, nvoxels_threshold=None, cut_coords=(6, 6, 6),
                       background=None, vmin=None, vmax=None, cmap="hot", title="RSA Results",
                       colorbar_label=None, plot_regions=True, plot_montage=True,
                       plot_glass=True, plot_surface=True, connectivity=1, show=True):
    """
    Plot RSA brain results using multiple visualization modes.

    Parameters
    ----------
    img : str or Niimg-like object
        NIfTI image of RSA results.

    threshold : float or None, default=0.01
        Display threshold passed to nilearn plotting functions.

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

    cut_coords : list or tuple, length 3, default=(6, 6, 6)
        Cut coordinates for montage plotting.

    background : Niimg-like object, str, or None, default=None
        Background image.

        If None, default backgrounds are used.

    vmin, vmax : float or None, default=None
        Color limits.

    cmap : str or matplotlib colormap, default="hot"
        Colormap.

    title : str or None, default="RSA Results"
        Plot title.

    colorbar_label : str or None, default=None
        Colorbar label.

    plot_regions : bool, default=True
        Whether to plot orthogonal regions.

    plot_montage : bool, default=True
        Whether to plot montage cuts.

    plot_glass : bool, default=True
        Whether to plot glass brain.

    plot_surface : bool, default=True
        Whether to plot surface maps.

    connectivity : int, default=1
        Connectivity used for cluster-size thresholding.

    show : bool, default=True
        Whether to show the plots immediately.

    Returns
    -------
    displays : dict
        Dictionary of display objects.
    """

    _, data, has_results = _prepare_brainrsa_img(
        img,
        nvoxels_threshold=nvoxels_threshold,
        connectivity=connectivity,
    )

    if not has_results:
        print("No valid results.")
        return None

    displays = {}

    if background is None:
        region_background = get_bg_ch2()
        montage_background = get_bg_ch2bet()
    else:
        region_background = background
        montage_background = background

    # Important:
    # Set show=False for subfunctions, and show once at the end.
    if plot_regions:
        displays["regions"] = plot_brainrsa_regions(
            img,
            threshold=threshold,
            nvoxels_threshold=nvoxels_threshold,
            background=region_background,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            colorbar_label=colorbar_label,
            connectivity=connectivity,
            show=False,
        )

    if plot_montage:
        displays["montage"] = plot_brainrsa_montage(
            img,
            threshold=threshold,
            nvoxels_threshold=nvoxels_threshold,
            cut_coords=cut_coords,
            background=montage_background,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            colorbar_label=colorbar_label,
            connectivity=connectivity,
            show=False,
        )

    if plot_glass:
        displays["glass"] = plot_brainrsa_glass(
            img,
            threshold=threshold,
            nvoxels_threshold=nvoxels_threshold,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            colorbar_label=colorbar_label,
            connectivity=connectivity,
            show=False,
        )

    if plot_surface:
        displays["surface"] = plot_brainrsa_surface(
            img,
            threshold=threshold,
            nvoxels_threshold=nvoxels_threshold,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            colorbar_label=colorbar_label,
            connectivity=connectivity,
            show=False,
        )

    _maybe_show(show)

    return displays