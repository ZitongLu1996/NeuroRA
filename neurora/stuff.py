# -*- coding: utf-8 -*-

"""
A module for simple but important utility functions.
"""

__author__ = "Zitong Lu"

import os
import numpy as np
import nibabel as nib
from skimage.measure import label
from tqdm import tqdm
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t
from scipy.ndimage import gaussian_filter1d, gaussian_filter

np.seterr(divide="ignore", invalid="ignore")

# Get package absolute path.
package_root = os.path.dirname(os.path.abspath(__file__))


def _resolve_n_jobs(n_jobs):
    """
    Resolve the number of worker threads or processes.

    Parameters
    ----------
    n_jobs : int
        Number of workers.

        If n_jobs=-1, all available CPU cores will be used.

    Returns
    -------
    n_jobs : int
        Resolved number of workers.
    """

    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs should be an integer.")

    if n_jobs == -1:
        return os.cpu_count() or 1

    if n_jobs < 1:
        raise ValueError("n_jobs should be a positive integer or -1.")

    return n_jobs


def _validate_backend(backend):
    """
    Validate joblib backend.

    Parameters
    ----------
    backend : str
        Joblib backend.

        Supported options are:

        - "threading"
        - "loky"

    Returns
    -------
    backend : str
        Validated joblib backend.
    """

    valid_backends = {"threading", "loky"}

    if backend not in valid_backends:
        raise ValueError(
            "backend should be one of {'threading', 'loky'}."
        )

    return backend


def _validate_parallel_args(parallel, n_jobs, backend):
    """
    Validate parallel computation arguments.

    Parameters
    ----------
    parallel : bool
        Whether to use joblib parallel computation.

    n_jobs : int
        Number of workers.

    backend : str
        Joblib backend.

    Returns
    -------
    n_jobs : int
        Resolved number of workers.

    backend : str
        Validated backend.
    """

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

    return n_jobs, backend


def _validate_3d_map(img, name="img"):
    """
    Validate a 3-D map.

    Parameters
    ----------
    img : array_like, shape [nx, ny, nz]
        Input 3-D map.

    name : str, default="img"
        Name used in error messages.

    Returns
    -------
    img : ndarray, shape [nx, ny, nz]
        Validated 3-D map.
    """

    img = np.asarray(img, dtype=float)

    if img.ndim != 3:
        raise ValueError(f"{name} should be a 3-D array with shape [nx, ny, nz].")

    return img


def _validate_p_threshold(p_threshold, name="p_threshold"):
    """
    Validate a p-value threshold.

    Parameters
    ----------
    p_threshold : float
        P-value threshold.

    name : str, default="p_threshold"
        Name used in error messages.

    Returns
    -------
    p_threshold : float
        Validated p-value threshold.
    """

    if not isinstance(p_threshold, (int, float)):
        raise TypeError(f"{name} should be a number.")

    p_threshold = float(p_threshold)

    if p_threshold <= 0 or p_threshold > 1:
        raise ValueError(f"{name} should be in the interval (0, 1].")

    return p_threshold


def _bh_fdr_correct(p_values):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array_like, shape [n_tests]
        P-values.

    Returns
    -------
    q_values : ndarray, shape [n_tests]
        FDR-adjusted p-values.
    """

    p_values = np.asarray(p_values, dtype=float)

    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)

    p = p_values[finite_mask]
    n_tests = p.size

    if n_tests == 0:
        return q_values

    order = np.argsort(p)
    ranked_p = p[order]

    ranked_q = ranked_p * n_tests / np.arange(1, n_tests + 1)

    # Enforce monotonicity.
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty_like(p, dtype=float)
    q[order] = ranked_q

    q_values[finite_mask] = q

    return q_values


def _get_cluster_sizes(binary_map, connectivity=1):
    """
    Label clusters in a binary map and return cluster sizes.

    Parameters
    ----------
    binary_map : array_like
        Binary cluster-forming map.

    connectivity : int, default=1
        Connectivity for cluster labeling.

    Returns
    -------
    labels : ndarray
        Labeled cluster map. Background is 0.

    cluster_sizes : ndarray, shape [n_clusters]
        Cluster sizes. cluster_sizes[i] corresponds to label i + 1.
    """

    binary_map = np.asarray(binary_map).astype(bool)

    if connectivity not in {1, 2, 3}:
        raise ValueError("connectivity should be 1, 2, or 3.")

    labels = label(binary_map, connectivity=connectivity)
    n_clusters = int(np.max(labels))

    if n_clusters == 0:
        return labels, np.array([], dtype=int)

    cluster_sizes = np.bincount(labels.ravel())[1:]

    return labels, cluster_sizes


def _compute_t_threshold(n_subs, p_threshold=0.05, tail=1):
    """
    Convert a p-value threshold to a t-value threshold.

    Parameters
    ----------
    n_subs : int
        Number of subjects.

    p_threshold : float, default=0.05
        Cluster-forming p-value threshold.

    tail : {-1, 0, 1}
        Tail of the test.

        - 1: greater
        - -1: less
        - 0: two-sided

    Returns
    -------
    threshold : float
        T-value threshold for MNE cluster test.
    """

    p_threshold = _validate_p_threshold(p_threshold, name="p_threshold")

    if tail not in {-1, 0, 1}:
        raise ValueError("tail should be -1, 0, or 1.")

    df = n_subs - 1

    if df < 1:
        raise ValueError("At least two subjects are required.")

    if tail == 0:
        threshold = t.ppf(1 - p_threshold / 2, df)
    else:
        threshold = t.ppf(1 - p_threshold, df)

    return threshold


def _clusters_to_mask(T_obs, clusters, cluster_p_values, clusterp_threshold, tail):
    """
    Convert MNE cluster results to a signed significance mask.

    Parameters
    ----------
    T_obs : ndarray
        Observed t-values.

    clusters : list
        Clusters returned by MNE.

    cluster_p_values : ndarray
        Cluster-level p-values returned by MNE.

    clusterp_threshold : float
        Cluster-level p-value threshold.

    tail : {-1, 0, 1}
        Tail of the test.

    Returns
    -------
    sig_mask : ndarray
        Signed significance mask.

        Values are:

        - 1 : significantly positive
        - -1 : significantly negative
        - 0 : not significant
    """

    clusterp_threshold = _validate_p_threshold(
        clusterp_threshold,
        name="clusterp_threshold",
    )

    sig_mask = np.zeros_like(T_obs, dtype=float)

    for cluster, cluster_p in zip(clusters, cluster_p_values):
        if cluster_p >= clusterp_threshold:
            continue

        if isinstance(cluster, tuple):
            cluster_mask = np.zeros_like(T_obs, dtype=bool)
            cluster_mask[cluster] = True
        else:
            cluster_mask = cluster

        mean_t = np.nanmean(T_obs[cluster_mask])

        if tail == 1:
            sig_mask[cluster_mask] = 1
        elif tail == -1:
            sig_mask[cluster_mask] = -1
        else:
            if mean_t > 0:
                sig_mask[cluster_mask] = 1
            elif mean_t < 0:
                sig_mask[cluster_mask] = -1

    return sig_mask


def _remove_small_clusters(sig_mask, n_threshold):
    """
    Remove significant clusters smaller than n_threshold.

    Parameters
    ----------
    sig_mask : ndarray
        Signed significance mask.

    n_threshold : int
        Minimum cluster size.

    Returns
    -------
    cleaned_mask : ndarray
        Cleaned significance mask.
    """

    if not isinstance(n_threshold, int) or n_threshold < 1:
        raise ValueError("n_threshold should be a positive integer.")

    cleaned_mask = np.array(sig_mask, copy=True)

    for sign in [1, -1]:
        labels = label(cleaned_mask == sign, connectivity=1)
        n_clusters = int(np.max(labels))

        for cluster_idx in range(1, n_clusters + 1):
            cluster_size = np.sum(labels == cluster_idx)

            if cluster_size < n_threshold:
                cleaned_mask[labels == cluster_idx] = 0

    return cleaned_mask


def _mne_cluster_1samp_wrapper(X, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                               n_iter=1000, tail=1, random_state=None, n_jobs=None):
    """
    Run MNE one-sample cluster permutation test and return a signed mask.

    Parameters
    ----------
    X : array_like, shape [n_subs, ...]
        Subject-level data.

    p_threshold : float, default=0.05
        Cluster-forming p-value threshold.

    clusterp_threshold : float, default=0.05
        Cluster-level p-value threshold.

    n_threshold : int, default=2
        Minimum cluster size.

    n_iter : int, default=1000
        Number of permutations.

    tail : {-1, 0, 1}, default=1
        Tail of the test.

        - 1: positive one-sided test
        - -1: negative one-sided test
        - 0: two-sided test

    random_state : int or None, default=None
        Random seed.

    n_jobs : int or None, default=None
        Number of parallel jobs for MNE.

    Returns
    -------
    sig_mask : ndarray, shape [...]
        Signed significance mask.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim < 2:
        raise ValueError("X should have shape [n_subs, ...].")

    n_subs = X.shape[0]

    if n_subs < 2:
        raise ValueError("At least two subjects are required.")

    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter should be a positive integer.")

    threshold = _compute_t_threshold(
        n_subs=n_subs,
        p_threshold=p_threshold,
        tail=tail,
    )

    T_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
        X,
        threshold=threshold,
        n_permutations=n_iter,
        tail=tail,
        seed=random_state,
        n_jobs=n_jobs,
        out_type="indices",
    )

    sig_mask = _clusters_to_mask(
        T_obs=T_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        clusterp_threshold=clusterp_threshold,
        tail=tail,
    )

    sig_mask = _remove_small_clusters(
        sig_mask=sig_mask,
        n_threshold=n_threshold,
    )

    return sig_mask


def limtozero(x, tol=1e-15):
    """
    Set values very close to zero to exactly zero.

    Parameters
    ----------
    x : float or array_like
        Input value or array.

    tol : float, default=1e-15
        Absolute tolerance for treating values as zero.

    Returns
    -------
    out : float or ndarray
        Value with near-zero values replaced by zero.
    """

    if tol < 0:
        raise ValueError("tol should be non-negative.")

    x_arr = np.asarray(x)

    out = np.where(np.abs(x_arr) < tol, 0, x_arr)

    if np.isscalar(x):
        return float(out)

    return out
    

def get_affine(file_name):
    """
    Get the affine matrix of a NIfTI image.

    Parameters
    ----------
    file_name : str
        Filename of a sample fMRI image.

    Returns
    -------
    affine : ndarray, shape [4, 4]
        Affine matrix of the image.
    """

    if not isinstance(file_name, str) or file_name == "":
        raise ValueError("file_name should be a non-empty string.")

    img = nib.load(file_name)

    return img.affine


def correct_by_threshold(img, threshold, connectivity=1):
    """
    Correct fMRI RSA results by removing clusters smaller than a voxel threshold.

    Parameters
    ----------
    img : array_like, shape [nx, ny, nz]
        A 3-D fMRI RSA result map.

    threshold : int
        Minimum cluster size in voxels.

        Clusters with fewer than threshold non-zero voxels will be removed.

    connectivity : {1, 2, 3}, default=1
        Cluster connectivity.

    Returns
    -------
    corrected_img : ndarray, shape [nx, ny, nz]
        Threshold-corrected image.
    """

    img = _validate_3d_map(img, name="img")

    if not isinstance(threshold, int) or threshold < 1:
        raise ValueError("threshold should be a positive integer.")

    corrected_img = np.array(img, copy=True)

    active_map = (corrected_img != 0) & np.isfinite(corrected_img)

    labels, cluster_sizes = _get_cluster_sizes(
        active_map,
        connectivity=connectivity,
    )

    for cluster_idx, cluster_size in enumerate(cluster_sizes, start=1):
        if cluster_size < threshold:
            corrected_img[labels == cluster_idx] = 0

    print("finished correction")

    return corrected_img


def fwe_correct(p, p_threshold):
    """
    FWE correction for fMRI RSA results using Bonferroni correction.

    Parameters
    ----------
    p : array_like, shape [nx, ny, nz]
        A 3-D p-value map.

    p_threshold : float
        Corrected p-value threshold.

    Returns
    -------
    fwep : ndarray, shape [nx, ny, nz]
        FWE-corrected p-value map.

        Voxels whose corrected p-values do not survive p_threshold are NaN.
    """

    p = _validate_3d_map(p, name="p")
    p_threshold = _validate_p_threshold(p_threshold, name="p_threshold")

    finite_mask = np.isfinite(p)
    n_tests = int(np.sum(finite_mask))

    if n_tests == 0:
        return np.full_like(p, np.nan, dtype=float)

    corrected_p = np.full_like(p, np.nan, dtype=float)
    corrected_p[finite_mask] = np.minimum(p[finite_mask] * n_tests, 1.0)

    fwep = np.full_like(p, np.nan, dtype=float)
    survive = finite_mask & (corrected_p < p_threshold)
    fwep[survive] = corrected_p[survive]

    print("finished FWE correct")

    return fwep


def fdr_correct(p, p_threshold):
    """
    FDR correction for fMRI RSA results using Benjamini-Hochberg correction.

    Parameters
    ----------
    p : array_like, shape [nx, ny, nz]
        A 3-D p-value map.

    p_threshold : float
        FDR-corrected p-value threshold.

    Returns
    -------
    fdrp : ndarray, shape [nx, ny, nz]
        FDR-corrected p-value map.

        Voxels whose corrected p-values do not survive p_threshold are NaN.
    """

    p = _validate_3d_map(p, name="p")
    p_threshold = _validate_p_threshold(p_threshold, name="p_threshold")

    q_flat = _bh_fdr_correct(p.ravel())
    q_map = q_flat.reshape(p.shape)

    fdrp = np.full_like(p, np.nan, dtype=float)
    survive = np.isfinite(q_map) & (q_map < p_threshold)
    fdrp[survive] = q_map[survive]

    print("finished FDR correct")

    return fdrp


def get_bg_ch2():
    """
    Get the absolute file path of ch2.nii.gz.

    Returns
    -------
    path : str
        Absolute file path of ch2.nii.gz.
    """

    return os.path.join(package_root, "template", "ch2.nii.gz")


def get_bg_ch2bet():
    """
    Get the absolute file path of ch2bet.nii.gz.

    Returns
    -------
    path : str
        Absolute file path of ch2bet.nii.gz.
    """

    return os.path.join(package_root, "template", "ch2bet.nii.gz")


def get_HOcort():
    """
    Get the absolute file path of HarvardOxford cortical atlas.

    Returns
    -------
    path : str
        Absolute file path of HarvardOxford-cort-maxprob-thr0-1mm.nii.gz.
    """

    return os.path.join(
        package_root,
        "template",
        "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz",
    )


def datamask(fmri_data, mask_data):
    """
    Filter a 3-D fMRI map by a 3-D ROI mask.

    Parameters
    ----------
    fmri_data : array_like, shape [nx, ny, nz]
        fMRI data.

    mask_data : array_like, shape [nx, ny, nz]
        ROI mask.

    Returns
    -------
    newfmri_data : ndarray, shape [nx, ny, nz]
        Masked fMRI data. Voxels outside the mask are NaN.
    """

    fmri_data = _validate_3d_map(fmri_data, name="fmri_data")
    mask_data = _validate_3d_map(mask_data, name="mask_data")

    if fmri_data.shape != mask_data.shape:
        raise ValueError("fmri_data and mask_data should have the same shape.")

    valid_mask = (mask_data != 0) & np.isfinite(mask_data)

    newfmri_data = np.full_like(fmri_data, np.nan, dtype=float)
    newfmri_data[valid_mask] = fmri_data[valid_mask]

    return newfmri_data


def position_to_mni(point, affine):
    """
    Project voxel coordinates to MNI/world coordinates.

    Parameters
    ----------
    point : array_like, shape [3]
        Voxel coordinate.

    affine : array_like, shape [4, 4]
        Affine matrix.

    Returns
    -------
    newpoint : ndarray, shape [3]
        MNI/world coordinate.
    """

    point = np.asarray(point, dtype=float)
    affine = np.asarray(affine, dtype=float)

    if point.shape != (3,):
        raise ValueError("point should have shape [3].")

    if affine.shape != (4, 4):
        raise ValueError("affine should have shape [4, 4].")

    return nib.affines.apply_affine(affine, point)


def mniposition_to(mnipoint, affine):
    """
    Project MNI/world coordinates to voxel coordinates.

    Parameters
    ----------
    mnipoint : array_like, shape [3]
        MNI/world coordinate.

    affine : array_like, shape [4, 4]
        Affine matrix.

    Returns
    -------
    voxel_index : tuple of int
        Voxel coordinate.
    """

    mnipoint = np.asarray(mnipoint, dtype=float)
    affine = np.asarray(affine, dtype=float)

    if mnipoint.shape != (3,):
        raise ValueError("mnipoint should have shape [3].")

    if affine.shape != (4, 4):
        raise ValueError("affine should have shape [4, 4].")

    inv_affine = np.linalg.inv(affine)
    voxel = nib.affines.apply_affine(inv_affine, mnipoint)

    voxel = np.round(voxel).astype(int)

    return tuple(voxel)


def mask_to(mask, size, affine, filename=None):
    """
    Convert a mask image to a target data template.

    Parameters
    ----------
    mask : str
        File path of the mask image.

    size : tuple or list of int, length 3
        Target image size [nx, ny, nz].

    affine : array_like, shape [4, 4]
        Target affine matrix.

    filename : str or None, default=None
        Output filename. If None, "newmask.nii" will be used.

    Returns
    -------
    filename : str
        Output filename of the generated mask image.
    """

    if not isinstance(mask, str) or mask == "":
        raise ValueError("mask should be a non-empty file path.")

    size = tuple(map(int, size))
    affine = np.asarray(affine, dtype=float)

    if len(size) != 3:
        raise ValueError("size should have length 3.")

    if min(size) < 1:
        raise ValueError("size should contain positive integers.")

    if affine.shape != (4, 4):
        raise ValueError("affine should have shape [4, 4].")

    mask_img = nib.load(mask)
    mask_data = mask_img.get_fdata()
    mask_affine = mask_img.affine

    newdata = np.zeros(size, dtype=float)

    source_indices = np.argwhere((mask_data != 0) & np.isfinite(mask_data))

    inv_target_affine = np.linalg.inv(affine)

    for source_index in source_indices:
        world_coord = nib.affines.apply_affine(mask_affine, source_index)
        target_index = nib.affines.apply_affine(inv_target_affine, world_coord)
        target_index = np.round(target_index).astype(int)

        x, y, z = target_index

        if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
            i, j, k = source_index
            newdata[x, y, z] = mask_data[i, j, k]

    if filename is None:
        filename = "newmask.nii"
    elif not filename.endswith(".nii") and not filename.endswith(".nii.gz"):
        filename = filename + ".nii"

    out_img = nib.Nifti1Image(newdata, affine)
    nib.save(out_img, filename)

    return filename


def clusterbased_permutation_1d_1samp_1sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                             n_threshold=2, n_iter=1000, random_state=None, n_jobs=None):
    """
    One-sample one-sided cluster-based permutation test for 1-D results.

    Parameters
    ----------
    results : array_like, shape [n_subs, x]
        Subject-level results.

    level : float, default=0
        Null-hypothesis level.

    p_threshold : float, default=0.05
        Cluster-forming p-value threshold.

    clusterp_threshold : float, default=0.05
        Cluster-level p-value threshold.

    n_threshold : int, default=2
        Minimum cluster size.

    n_iter : int, default=1000
        Number of permutations.

    random_state : int or None, default=None
        Random seed.

    n_jobs : int or None, default=None
        Number of parallel jobs for MNE.

    Returns
    -------
    ps : ndarray, shape [x]
        Binary significance vector.
    """

    results = np.asarray(results, dtype=float)

    if results.ndim != 2:
        raise ValueError("results should have shape [n_subs, x].")

    X = results - level

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=1,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_1d_1samp_2sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                             n_threshold=2, n_iter=1000, random_state=None, n_jobs=None):
    """
    One-sample two-sided cluster-based permutation test for 1-D results.

    Returns
    -------
    ps : ndarray, shape [x]
        Signed significance vector.

        - 1 : significantly greater than level
        - -1 : significantly less than level
        - 0 : not significant
    """

    results = np.asarray(results, dtype=float)

    if results.ndim != 2:
        raise ValueError("results should have shape [n_subs, x].")

    X = results - level

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_1d_1sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05,
                                       n_threshold=2, n_iter=1000, random_state=None, n_jobs=None):
    """
    Paired one-sided cluster-based permutation test for 1-D results.

    Tests whether results1 > results2.
    """

    results1 = np.asarray(results1, dtype=float)
    results2 = np.asarray(results2, dtype=float)

    if results1.shape != results2.shape or results1.ndim != 2:
        raise ValueError(
            "results1 and results2 should have the same shape [n_subs, x]."
        )

    X = results1 - results2

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=1,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_1d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05,
                                       n_threshold=2, n_iter=1000, random_state=None, n_jobs=None):
    """
    Paired two-sided cluster-based permutation test for 1-D results.
    """

    results1 = np.asarray(results1, dtype=float)
    results2 = np.asarray(results2, dtype=float)

    if results1.shape != results2.shape or results1.ndim != 2:
        raise ValueError(
            "results1 and results2 should have the same shape [n_subs, x]."
        )

    X = results1 - results2

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_2d_1samp_1sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                             n_threshold=4, n_iter=1000, random_state=None, n_jobs=None):
    """
    One-sample one-sided cluster-based permutation test for 2-D results.
    """

    results = np.asarray(results, dtype=float)

    if results.ndim != 3:
        raise ValueError("results should have shape [n_subs, x1, x2].")

    X = results - level

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=1,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_2d_1samp_2sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                             n_threshold=4, n_iter=1000, random_state=None, n_jobs=None):
    """
    One-sample two-sided cluster-based permutation test for 2-D results.
    """

    results = np.asarray(results, dtype=float)

    if results.ndim != 3:
        raise ValueError("results should have shape [n_subs, x1, x2].")

    X = results - level

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_2d_1sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05,
                                       n_threshold=4, n_iter=1000, random_state=None, n_jobs=None):
    """
    Paired one-sided cluster-based permutation test for 2-D results.

    Tests whether results1 > results2.
    """

    results1 = np.asarray(results1, dtype=float)
    results2 = np.asarray(results2, dtype=float)

    if results1.shape != results2.shape or results1.ndim != 3:
        raise ValueError(
            "results1 and results2 should have the same shape [n_subs, x1, x2]."
        )

    X = results1 - results2

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=1,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def clusterbased_permutation_2d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05,
                                       n_threshold=4, n_iter=1000, random_state=None, n_jobs=None):
    """
    Paired two-sided cluster-based permutation test for 2-D results.
    """

    results1 = np.asarray(results1, dtype=float)
    results2 = np.asarray(results2, dtype=float)

    if results1.shape != results2.shape or results1.ndim != 3:
        raise ValueError(
            "results1 and results2 should have the same shape [n_subs, x1, x2]."
        )

    X = results1 - results2

    return _mne_cluster_1samp_wrapper(
        X=X,
        p_threshold=p_threshold,
        clusterp_threshold=clusterp_threshold,
        n_threshold=n_threshold,
        n_iter=n_iter,
        tail=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def smooth_1d_gaussian(data, smooth_window=5):
    """
    Smooth 1-D time-series data using a Gaussian filter.

    Parameters
    ----------
    data : array_like, shape [n_timepoints]
        Input 1-D time-series data.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    Returns
    -------
    smoothed_data : ndarray, shape [n_timepoints]
        Smoothed 1-D data.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 1:
        raise ValueError("data should be a 1-D array with shape [n_timepoints].")

    if not isinstance(smooth_window, int) or smooth_window < 1:
        raise ValueError("smooth_window should be a positive integer.")

    if smooth_window == 1:
        return np.array(data, copy=True)

    sigma = smooth_window / 6.0

    smoothed_data = gaussian_filter1d(
        data,
        sigma=sigma,
        mode="reflect",
        truncate=3.0,
    )

    return smoothed_data


def smooth_2d_gaussian(data, smooth_window=5):
    """
    Smooth 2-D data using a Gaussian filter.

    Parameters
    ----------
    data : array_like, shape [n_timepoints1, n_timepoints2]
        Input 2-D data.

    smooth_window : int, default=5
        Approximate smoothing window in time points.

    Returns
    -------
    smoothed_data : ndarray, shape [n_timepoints1, n_timepoints2]
        Smoothed 2-D data.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            "data should be a 2-D array with shape [n_timepoints1, n_timepoints2]."
        )

    if not isinstance(smooth_window, int) or smooth_window < 1:
        raise ValueError("smooth_window should be a positive integer.")

    if smooth_window == 1:
        return np.array(data, copy=True)

    sigma = smooth_window / 6.0

    smoothed_data = gaussian_filter(
        data,
        sigma=sigma,
        mode="reflect",
        truncate=3.0,
    )

    return smoothed_data


def normalize_rdm(rdm, method="minmax", keep_diagonal=True):
    """
    Normalize a representational dissimilarity matrix (RDM).

    Parameters
    ----------
    rdm : array_like, shape [n_cons, n_cons] or [..., n_cons, n_cons]
        Input RDM or a stack of RDMs.

        The last two dimensions should be square matrices.

    method : {"minmax", "zscore"}, default="minmax"
        Normalization method.

        - "minmax": scale off-diagonal values to the range [0, 1]
        - "zscore": standardize off-diagonal values to mean 0 and standard deviation 1

    keep_diagonal : bool, default=True
        Whether to keep the diagonal values unchanged.

        If True, diagonal values are copied from the original RDM.
        If False, diagonal values are set to 0.

    Returns
    -------
    normalized_rdm : ndarray
        Normalized RDM with the same shape as the input.

    Notes
    -----
    Normalization is applied separately to each RDM when a stack of RDMs is provided.

    Only off-diagonal values are used to estimate min/max or mean/std.
    """

    rdm = np.asarray(rdm, dtype=float)

    if rdm.ndim < 2:
        raise ValueError(
            "rdm should have shape [n_cons, n_cons] or [..., n_cons, n_cons]."
        )

    if rdm.shape[-1] != rdm.shape[-2]:
        raise ValueError(
            "The last two dimensions of rdm should be square: [n_cons, n_cons]."
        )

    if method not in {"minmax", "zscore"}:
        raise ValueError("method should be one of {'minmax', 'zscore'}.")

    if not isinstance(keep_diagonal, bool):
        raise TypeError("keep_diagonal should be True or False.")

    n_cons = rdm.shape[-1]
    triu_idx = np.triu_indices(n_cons, k=1)

    normalized_rdm = np.full_like(rdm, np.nan, dtype=float)

    leading_shape = rdm.shape[:-2]

    if leading_shape == ():
        rdm_indices = [()]
    else:
        rdm_indices = list(np.ndindex(leading_shape))

    for idx in rdm_indices:
        if idx == ():
            this_rdm = rdm
            out_rdm = normalized_rdm
        else:
            this_rdm = rdm[idx]
            out_rdm = normalized_rdm[idx]

        values = this_rdm[triu_idx]
        valid_values = values[np.isfinite(values)]

        if valid_values.size == 0:
            continue

        if method == "minmax":
            vmin = np.min(valid_values)
            vmax = np.max(valid_values)

            if vmax == vmin:
                normalized_values = np.zeros_like(values, dtype=float)
                normalized_values[np.isfinite(values)] = 0
            else:
                normalized_values = (values - vmin) / (vmax - vmin)

        else:
            mean = np.mean(valid_values)
            std = np.std(valid_values)

            if std == 0:
                normalized_values = np.zeros_like(values, dtype=float)
                normalized_values[np.isfinite(values)] = 0
            else:
                normalized_values = (values - mean) / std

        out_rdm[:, :] = np.nan

        out_rdm[triu_idx] = normalized_values
        out_rdm[(triu_idx[1], triu_idx[0])] = normalized_values

        if keep_diagonal:
            diag_idx = np.diag_indices(n_cons)
            out_rdm[diag_idx] = this_rdm[diag_idx]
        else:
            np.fill_diagonal(out_rdm, 0)

    return normalized_rdm