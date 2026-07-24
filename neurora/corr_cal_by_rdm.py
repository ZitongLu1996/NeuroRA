# -*- coding: utf-8 -*-

' a module for calculating the Similarity (Spearman Correlation Cosfficient) between model RDMs and neural RDMs '

__author__ = 'Zitong Lu'

import os
import numpy as np
import pandas as pd
import pingouin as pg
from joblib import Parallel, delayed
from neurora.stuff import _resolve_n_jobs, _validate_backend
from neurora.rdm_corr import rdm_correlation_spearman
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')


def _validate_model_rdms(model_rdms):
    """
    Validate model RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    Returns
    -------
    model_rdms : ndarray, shape [n_models, n_cons, n_cons]
        Validated model RDMs.
    n_models : int
        Number of model RDMs.
    n_cons : int
        Number of conditions.
    """

    model_rdms = np.asarray(model_rdms)

    if model_rdms.ndim != 3:
        raise ValueError(
            "The shape of model_rdms should be [n_models, n_cons, n_cons]."
        )

    if model_rdms.shape[1] != model_rdms.shape[2]:
        raise ValueError(
            "Each model RDM should be a square matrix with shape [n_cons, n_cons]."
        )

    n_models, n_cons, _ = model_rdms.shape

    return model_rdms, n_models, n_cons


def _vectorize_model_rdms(model_rdms):
    """
    Vectorize model RDMs using the upper triangle excluding the diagonal.

    Parameters
    ----------
    model_rdms : ndarray, shape [n_models, n_cons, n_cons]
        Model RDMs.

    Returns
    -------
    model_vecs : ndarray, shape [n_pairs, n_models]
        Vectorized model RDMs.

    triu_idx : tuple of ndarray
        Upper-triangle indices used for vectorization.
    """

    model_rdms, n_models, n_cons = _validate_model_rdms(model_rdms)

    triu_idx = np.triu_indices(n_cons, k=1)
    model_vecs = model_rdms[:, triu_idx[0], triu_idx[1]].T

    return model_vecs, triu_idx


def _permute_rdm_labels(rdm, permutation):
    """
    Permute condition labels of one RDM by applying the same permutation
    to rows and columns.

    Parameters
    ----------
    rdm : ndarray, shape [n_cons, n_cons]
        One RDM.

    permutation : ndarray, shape [n_cons]
        Permuted condition indices.

    Returns
    -------
    permuted_rdm : ndarray, shape [n_cons, n_cons]
        Label-permuted RDM.
    """

    rdm = np.asarray(rdm)

    return rdm[np.ix_(permutation, permutation)]


def _make_permutations(n_cons, n_permutation, random_state=None):
    """
    Generate condition-label permutations.

    Parameters
    ----------
    n_cons : int
        Number of conditions.

    n_permutation : int
        Number of permutations.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    permutations : ndarray, shape [n_permutation, n_cons]
        Permutation indices.
    """

    if not isinstance(n_permutation, int) or n_permutation < 1:
        raise ValueError("n_permutation should be a positive integer.")

    rng = np.random.default_rng(random_state)

    permutations = np.empty((n_permutation, n_cons), dtype=int)

    for i in range(n_permutation):
        permutations[i] = rng.permutation(n_cons)

    return permutations


def _compute_single_target(model_rdms, target_rdm):
    """
    Compute Spearman correlations between all model RDMs and one target RDM.

    Parameters
    ----------
    model_rdms : ndarray, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdm : ndarray, shape [n_cons, n_cons]
        One target RDM.

    Returns
    -------
    single_corrs : ndarray, shape [n_models, 2]
        Spearman correlation results for one target RDM.

        The last dimension contains:

        - single_corrs[:, 0] : Spearman r
        - single_corrs[:, 1] : p-value
    """

    n_models = model_rdms.shape[0]
    single_corrs = np.full((n_models, 2), np.nan)

    for m in range(n_models):
        single_corrs[m] = rdm_correlation_spearman(
            model_rdms[m],
            target_rdm,
        )

    return single_corrs


def _compute_single_target_partialcorr(model_vecs, target_rdm, triu_idx):
    """
    Compute Spearman partial correlations between all model RDMs and one target RDM.

    For each model RDM, the partial correlation is computed between the
    target RDM and the current model RDM while controlling for all other
    model RDMs.

    Parameters
    ----------
    model_vecs : ndarray, shape [n_pairs, n_models]
        Vectorized model RDMs.

    target_rdm : ndarray, shape [n_cons, n_cons]
        One target RDM.

    triu_idx : tuple of ndarray
        Upper-triangle indices used for vectorizing target_rdm.

    Returns
    -------
    single_corrs : ndarray, shape [n_models, 2]
        Spearman partial correlation results for one target RDM.

        The last dimension contains:

        - single_corrs[:, 0] : partial correlation coefficient
        - single_corrs[:, 1] : p-value
    """

    n_models = model_vecs.shape[1]
    model_cols = [f"model_{m}" for m in range(n_models)]

    target_vec = np.asarray(target_rdm)[triu_idx]

    df = pd.DataFrame(model_vecs, columns=model_cols)
    df["target"] = target_vec

    # Remove rows with NaN or infinite values in either model or target vectors.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    single_corrs = np.full((n_models, 2), np.nan, dtype=float)

    for m, model_col in enumerate(model_cols):
        covars = [col for col in model_cols if col != model_col]

        try:
            if len(covars) == 0:
                # If there is only one model RDM, partial correlation reduces
                # to a regular Spearman correlation.
                result = pg.corr(
                    x=df[model_col],
                    y=df["target"],
                    method="spearman",
                )
            else:
                # Need enough observations relative to the number of covariates.
                if df.shape[0] <= len(covars) + 2:
                    continue

                result = pg.partial_corr(
                    data=df,
                    x=model_col,
                    y="target",
                    covar=covars,
                    method="spearman",
                )

            single_corrs[m, 0] = result["r"].iloc[0]
            single_corrs[m, 1] = result["p-val"].iloc[0]

        except Exception:
            # Constant vectors, singular covariance, or insufficient valid data
            # can make partial correlation undefined.
            single_corrs[m] = np.nan

    return single_corrs


def _compute_spearman_corrs(model_rdms, target_rdms, desc="Calculating",
                            parallel=False, n_jobs=-1, backend="loky"):
    """
    Compute Spearman correlations between multiple model RDMs and target RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdms : array_like, shape [..., n_cons, n_cons]
        Target RDMs. The leading dimensions can be arbitrary.

    desc : str, default="Calculating"
        Description shown in the tqdm progress bar.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

        - "threading" uses multiple threads and shares memory within one process.
        - "loky" uses multiple processes and can bypass the Python GIL.
    Returns
    -------
    corrs : ndarray, shape [..., n_models, 2]
        Spearman correlation results.

        The last dimension contains:

        - corrs[..., 0] : Spearman r
        - corrs[..., 1] : p-value
    """

    model_rdms, n_models, n_cons = _validate_model_rdms(model_rdms)
    target_rdms = np.asarray(target_rdms)

    if target_rdms.ndim < 2:
        raise ValueError(
            "target_rdms should have shape [n_cons, n_cons] or [..., n_cons, n_cons]."
        )

    if target_rdms.shape[-2:] != (n_cons, n_cons):
        raise ValueError(
            "The last two dimensions of target_rdms should match the model RDM shape: "
            "[n_cons, n_cons]."
        )

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

    leading_shape = target_rdms.shape[:-2]

    # Output shape:
    # target leading dimensions + model dimension + [r, p]
    corrs = np.full(leading_shape + (n_models, 2), np.nan)

    if leading_shape == ():
        target_indices = [()]
    else:
        target_indices = list(np.ndindex(leading_shape))

    def get_target_rdm(target_idx):
        if target_idx == ():
            return target_rdms
        return target_rdms[target_idx]

    if not parallel:
        for target_idx in tqdm(target_indices, desc=desc):
            target_rdm = get_target_rdm(target_idx)

            corrs[target_idx] = _compute_single_target(
                model_rdms=model_rdms,
                target_rdm=target_rdm,
            )

        return corrs

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
    )(
        delayed(_compute_single_target)(
            model_rdms,
            get_target_rdm(target_idx),
        )
        for target_idx in tqdm(target_indices, desc=desc)
    )

    for target_idx, single_corrs in zip(target_indices, results):
        corrs[target_idx] = single_corrs

    return corrs


def _compute_partialcorrs(model_rdms, target_rdms, desc="Calculating",
                          parallel=False, n_jobs=-1, backend="loky"):
    """
    Compute Spearman partial correlations between multiple model RDMs and target RDMs.

    For each model RDM, the partial correlation is computed between the
    target RDM and the current model RDM while controlling for all other
    model RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdms : array_like, shape [..., n_cons, n_cons]
        Target RDMs. The leading dimensions can be arbitrary.

    desc : str, default="Calculating"
        Description shown in the tqdm progress bar.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    corrs : ndarray, shape [..., n_models, 2]
        Spearman partial correlation results.

        The last dimension contains:

        - corrs[..., 0] : partial correlation coefficient
        - corrs[..., 1] : p-value
    """

    model_rdms, n_models, n_cons = _validate_model_rdms(model_rdms)
    target_rdms = np.asarray(target_rdms)

    if target_rdms.ndim < 2:
        raise ValueError(
            "target_rdms should have shape [n_cons, n_cons] or [..., n_cons, n_cons]."
        )

    if target_rdms.shape[-2:] != (n_cons, n_cons):
        raise ValueError(
            "The last two dimensions of target_rdms should match the model RDM shape: "
            "[n_cons, n_cons]."
        )

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

    model_vecs, triu_idx = _vectorize_model_rdms(model_rdms)

    leading_shape = target_rdms.shape[:-2]

    # Output shape:
    # target leading dimensions + model dimension + [partial r, p]
    corrs = np.full(leading_shape + (n_models, 2), np.nan, dtype=float)

    if leading_shape == ():
        target_indices = [()]
    else:
        target_indices = list(np.ndindex(leading_shape))

    def get_target_rdm(target_idx):
        if target_idx == ():
            return target_rdms
        return target_rdms[target_idx]

    if not parallel:
        for target_idx in tqdm(target_indices, desc=desc):
            target_rdm = get_target_rdm(target_idx)

            corrs[target_idx] = _compute_single_target_partialcorr(
                model_vecs=model_vecs,
                target_rdm=target_rdm,
                triu_idx=triu_idx,
            )

        return corrs

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
    )(
        delayed(_compute_single_target_partialcorr)(
            model_vecs,
            get_target_rdm(target_idx),
            triu_idx,
        )
        for target_idx in tqdm(target_indices, desc=desc)
    )

    for target_idx, single_corrs in zip(target_indices, results):
        corrs[target_idx] = single_corrs

    return corrs


def _compute_single_target_corr_perm(model_rdms, target_rdm, permutations):
    """
    Compute Spearman correlations between model RDMs and permuted target RDMs.

    Parameters
    ----------
    model_rdms : ndarray, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdm : ndarray, shape [n_cons, n_cons]
        One target RDM.

    permutations : ndarray, shape [n_permutation, n_cons]
        Condition-label permutations.

    Returns
    -------
    perm_corrs : ndarray, shape [n_permutation, n_models, 2]
        Correlation results for permuted target RDMs.
    """

    n_permutation = permutations.shape[0]
    n_models = model_rdms.shape[0]

    perm_corrs = np.full((n_permutation, n_models, 2), np.nan, dtype=float)

    for p, permutation in enumerate(permutations):
        permuted_rdm = _permute_rdm_labels(target_rdm, permutation)

        perm_corrs[p] = _compute_single_target(
            model_rdms=model_rdms,
            target_rdm=permuted_rdm,
        )

    return perm_corrs


def _compute_single_target_partialcorr_perm(model_vecs, target_rdm, triu_idx, permutations):
    """
    Compute Spearman partial correlations between model RDMs and permuted target RDMs.

    Parameters
    ----------
    model_vecs : ndarray, shape [n_pairs, n_models]
        Vectorized model RDMs.

    target_rdm : ndarray, shape [n_cons, n_cons]
        One target RDM.

    triu_idx : tuple of ndarray
        Upper-triangle indices.

    permutations : ndarray, shape [n_permutation, n_cons]
        Condition-label permutations.

    Returns
    -------
    perm_corrs : ndarray, shape [n_permutation, n_models, 2]
        Partial correlation results for permuted target RDMs.
    """

    n_permutation = permutations.shape[0]
    n_models = model_vecs.shape[1]

    perm_corrs = np.full((n_permutation, n_models, 2), np.nan, dtype=float)

    for p, permutation in enumerate(permutations):
        permuted_rdm = _permute_rdm_labels(target_rdm, permutation)

        perm_corrs[p] = _compute_single_target_partialcorr(
            model_vecs=model_vecs,
            target_rdm=permuted_rdm,
            triu_idx=triu_idx,
        )

    return perm_corrs


def _compute_spearman_corrs_perm(model_rdms, target_rdms, n_permutation=1000, desc="Calculating",
                                 parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Compute permutation-based Spearman correlations between model RDMs and target RDMs.

    The permutation is performed by shuffling the condition labels of each
    target RDM, that is, applying the same permutation to rows and columns
    of the target RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdms : array_like, shape [..., n_cons, n_cons]
        Target neural RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    desc : str, default="Calculating"
        Description shown in the tqdm progress bar.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [..., n_permutation, n_models, 2]
        Permutation correlation results.

        The last dimension contains:

        - perm_corrs[..., 0] : Spearman r
        - perm_corrs[..., 1] : p-value
    """

    model_rdms, n_models, n_cons = _validate_model_rdms(model_rdms)
    target_rdms = np.asarray(target_rdms)

    if target_rdms.ndim < 2:
        raise ValueError(
            "target_rdms should have shape [n_cons, n_cons] or [..., n_cons, n_cons]."
        )

    if target_rdms.shape[-2:] != (n_cons, n_cons):
        raise ValueError(
            "The last two dimensions of target_rdms should match the model RDM shape: "
            "[n_cons, n_cons]."
        )

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

    permutations = _make_permutations(
        n_cons=n_cons,
        n_permutation=n_permutation,
        random_state=random_state,
    )

    leading_shape = target_rdms.shape[:-2]

    perm_corrs = np.full(
        leading_shape + (n_permutation, n_models, 2),
        np.nan,
        dtype=float,
    )

    if leading_shape == ():
        target_indices = [()]
    else:
        target_indices = list(np.ndindex(leading_shape))

    def get_target_rdm(target_idx):
        if target_idx == ():
            return target_rdms
        return target_rdms[target_idx]

    if not parallel:
        for target_idx in tqdm(target_indices, desc=desc):
            target_rdm = get_target_rdm(target_idx)

            perm_corrs[target_idx] = _compute_single_target_corr_perm(
                model_rdms=model_rdms,
                target_rdm=target_rdm,
                permutations=permutations,
            )

        return perm_corrs

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
    )(
        delayed(_compute_single_target_corr_perm)(
            model_rdms,
            get_target_rdm(target_idx),
            permutations,
        )
        for target_idx in tqdm(target_indices, desc=desc)
    )

    for target_idx, target_perm_corrs in zip(target_indices, results):
        perm_corrs[target_idx] = target_perm_corrs

    return perm_corrs


def _compute_partialcorrs_perm(model_rdms, target_rdms, n_permutation=1000, desc="Calculating",
                               parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Compute permutation-based Spearman partial correlations between model RDMs and target RDMs.

    The permutation is performed by shuffling the condition labels of each
    target RDM, that is, applying the same permutation to rows and columns
    of the target RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    target_rdms : array_like, shape [..., n_cons, n_cons]
        Target neural RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    desc : str, default="Calculating"
        Description shown in the tqdm progress bar.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [..., n_permutation, n_models, 2]
        Permutation partial correlation results.

        The last dimension contains:

        - perm_corrs[..., 0] : Spearman partial correlation coefficient
        - perm_corrs[..., 1] : p-value
    """

    model_rdms, n_models, n_cons = _validate_model_rdms(model_rdms)
    target_rdms = np.asarray(target_rdms)

    if target_rdms.ndim < 2:
        raise ValueError(
            "target_rdms should have shape [n_cons, n_cons] or [..., n_cons, n_cons]."
        )

    if target_rdms.shape[-2:] != (n_cons, n_cons):
        raise ValueError(
            "The last two dimensions of target_rdms should match the model RDM shape: "
            "[n_cons, n_cons]."
        )

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

    model_vecs, triu_idx = _vectorize_model_rdms(model_rdms)

    permutations = _make_permutations(
        n_cons=n_cons,
        n_permutation=n_permutation,
        random_state=random_state,
    )

    leading_shape = target_rdms.shape[:-2]

    perm_corrs = np.full(
        leading_shape + (n_permutation, n_models, 2),
        np.nan,
        dtype=float,
    )

    if leading_shape == ():
        target_indices = [()]
    else:
        target_indices = list(np.ndindex(leading_shape))

    def get_target_rdm(target_idx):
        if target_idx == ():
            return target_rdms
        return target_rdms[target_idx]

    if not parallel:
        for target_idx in tqdm(target_indices, desc=desc):
            target_rdm = get_target_rdm(target_idx)

            perm_corrs[target_idx] = _compute_single_target_partialcorr_perm(
                model_vecs=model_vecs,
                target_rdm=target_rdm,
                triu_idx=triu_idx,
                permutations=permutations,
            )

        return perm_corrs

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
    )(
        delayed(_compute_single_target_partialcorr_perm)(
            model_vecs,
            get_target_rdm(target_idx),
            triu_idx,
            permutations,
        )
        for target_idx in tqdm(target_indices, desc=desc)
    )

    for target_idx, target_perm_corrs in zip(target_indices, results):
        perm_corrs[target_idx] = target_perm_corrs

    return perm_corrs


def rdms_corr(model_rdms, eeg_rdms, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate Spearman correlations between multiple model RDMs and EEG-like RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

        Each model RDM can be a behavioral RDM, a hypothesis-based coding
        model RDM, or a computational model RDM.

    eeg_rdms : array_like
        EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDMs.

        Supported shapes are:

        - [n_cons, n_cons]
        - [n1, n_cons, n_cons]
        - [n1, n2, n_cons, n_cons]
        - [n1, n2, n3, n_cons, n_cons]

        The leading dimensions can represent time windows, channels, subjects,
        or other analysis dimensions.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across EEG-like RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

        Use "threading" for shared-memory parallelism and lower memory overhead.
        Use "loky" for process-based parallelism when CPU-bound Python code is
        the main bottleneck.

    Returns
    -------
    corrs : ndarray
        Spearman correlation results between each EEG-like RDM and each model RDM.

        If eeg_rdms has shape [n_cons, n_cons], corrs has shape:

        - [n_models, 2]

        If eeg_rdms has shape [n1, n_cons, n_cons], corrs has shape:

        - [n1, n_models, 2]

        If eeg_rdms has shape [n1, n2, n_cons, n_cons], corrs has shape:

        - [n1, n2, n_models, 2]

        If eeg_rdms has shape [n1, n2, n3, n_cons, n_cons], corrs has shape:

        - [n1, n2, n3, n_models, 2]

        The last dimension contains:

        - corrs[..., 0] : Spearman r
        - corrs[..., 1] : p-value
    """

    eeg_rdms = np.asarray(eeg_rdms)

    if eeg_rdms.ndim < 2 or eeg_rdms.ndim > 5:
        raise ValueError(
            "The shape of eeg_rdms should be [n_cons, n_cons], "
            "[n1, n_cons, n_cons], [n1, n2, n_cons, n_cons], or "
            "[n1, n2, n3, n_cons, n_cons]."
        )

    if eeg_rdms.shape[-1] != eeg_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of eeg_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing Spearman correlations between model RDMs and EEG-like RDMs")

    corrs = _compute_spearman_corrs(
        model_rdms=model_rdms,
        target_rdms=eeg_rdms,
        desc="Calculating Spearman correlations between RDMs",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
    )

    print("\nComputing finished!")

    return corrs


def fmrirdms_corr(model_rdms, fmri_rdms, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate Spearman correlations between multiple model RDMs and fMRI searchlight RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

        Each model RDM can be a behavioral RDM, a hypothesis-based coding
        model RDM, or a computational model RDM.

    fmri_rdms : array_like, shape [n_x, n_y, n_z, n_cons, n_cons]
        fMRI searchlight RDMs.

        n_x, n_y, and n_z represent the number of searchlight units along
        the x, y, and z axes.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across fMRI searchlight RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

        Use "threading" for shared-memory parallelism and lower memory overhead.
        Use "loky" for process-based parallelism when CPU-bound Python code is
        the main bottleneck.

    Returns
    -------
    corrs : ndarray, shape [n_x, n_y, n_z, n_models, 2]
        Spearman correlation results between each fMRI searchlight RDM and
        each model RDM.

        The last dimension contains:

        - corrs[..., 0] : Spearman r
        - corrs[..., 1] : p-value
    """

    fmri_rdms = np.asarray(fmri_rdms)

    if fmri_rdms.ndim != 5:
        raise ValueError(
            "The shape of fmri_rdms should be [n_x, n_y, n_z, n_cons, n_cons]."
        )

    if fmri_rdms.shape[-1] != fmri_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of fmri_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing Spearman correlations between model RDMs and fMRI searchlight RDMs")

    corrs = _compute_spearman_corrs(
        model_rdms=model_rdms,
        target_rdms=fmri_rdms,
        desc="Calculating Spearman correlations between RDMs",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
    )

    print("\nComputing finished!")

    return corrs


def rdms_partialcorr(model_rdms, eeg_rdms, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate Spearman partial correlations between multiple model RDMs and EEG-like RDMs.

    For each model RDM, the partial correlation is computed between the
    EEG-like RDM and the current model RDM while controlling for all other
    model RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

        Each model RDM can be a behavioral RDM, a hypothesis-based coding
        model RDM, or a computational model RDM.

    eeg_rdms : array_like
        EEG/MEG/fNIRS/ECoG/sEEG/electrophysiological RDMs.

        Supported shapes are:

        - [n_cons, n_cons]
        - [n1, n_cons, n_cons]
        - [n1, n2, n_cons, n_cons]
        - [n1, n2, n3, n_cons, n_cons]

        The leading dimensions can represent time windows, channels, subjects,
        or other analysis dimensions.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across EEG-like RDMs.

        Parallelization is performed across target RDMs. For typical EEG-like
        inputs, this means parallelizing across time windows or across the
        combination of leading dimensions.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    corrs : ndarray
        Spearman partial correlation results between each EEG-like RDM and
        each model RDM while controlling for the other model RDMs.

        If eeg_rdms has shape [n_cons, n_cons], corrs has shape:

        - [n_models, 2]

        If eeg_rdms has shape [n1, n_cons, n_cons], corrs has shape:

        - [n1, n_models, 2]

        If eeg_rdms has shape [n1, n2, n_cons, n_cons], corrs has shape:

        - [n1, n2, n_models, 2]

        If eeg_rdms has shape [n1, n2, n3, n_cons, n_cons], corrs has shape:

        - [n1, n2, n3, n_models, 2]

        The last dimension contains:

        - corrs[..., 0] : Spearman partial correlation coefficient
        - corrs[..., 1] : p-value
    """

    eeg_rdms = np.asarray(eeg_rdms)

    if eeg_rdms.ndim < 2 or eeg_rdms.ndim > 5:
        raise ValueError(
            "The shape of eeg_rdms should be [n_cons, n_cons], "
            "[n1, n_cons, n_cons], [n1, n2, n_cons, n_cons], or "
            "[n1, n2, n3, n_cons, n_cons]."
        )

    if eeg_rdms.shape[-1] != eeg_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of eeg_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing Spearman partial correlations between model RDMs and EEG-like RDMs")

    corrs = _compute_partialcorrs(
        model_rdms=model_rdms,
        target_rdms=eeg_rdms,
        desc="Calculating Spearman partial correlations between RDMs",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
    )

    print("\nComputing finished!")

    return corrs


def fmrirdms_partialcorr(model_rdms, fmri_rdms, parallel=False, n_jobs=-1, backend="loky"):
    """
    Calculate Spearman partial correlations between multiple model RDMs and fMRI searchlight RDMs.

    For each model RDM, the partial correlation is computed between the
    fMRI searchlight RDM and the current model RDM while controlling for
    all other model RDMs.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

        Each model RDM can be a behavioral RDM, a hypothesis-based coding
        model RDM, or a computational model RDM.

    fmri_rdms : array_like, shape [n_x, n_y, n_z, n_cons, n_cons]
        fMRI searchlight RDMs.

        n_x, n_y, and n_z represent the number of searchlight units along
        the x, y, and z axes.

    parallel : bool, default=False
        Whether to use joblib-based parallel computation across fMRI
        searchlight RDMs.

        Parallelization is performed across searchlight units.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

        If n_jobs=-1, all available CPU cores will be used.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    Returns
    -------
    corrs : ndarray, shape [n_x, n_y, n_z, n_models, 2]
        Spearman partial correlation results between each fMRI searchlight RDM
        and each model RDM while controlling for the other model RDMs.

        The last dimension contains:

        - corrs[..., 0] : Spearman partial correlation coefficient
        - corrs[..., 1] : p-value
    """

    fmri_rdms = np.asarray(fmri_rdms)

    if fmri_rdms.ndim != 5:
        raise ValueError(
            "The shape of fmri_rdms should be [n_x, n_y, n_z, n_cons, n_cons]."
        )

    if fmri_rdms.shape[-1] != fmri_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of fmri_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing Spearman partial correlations between model RDMs and fMRI searchlight RDMs")

    corrs = _compute_partialcorrs(
        model_rdms=model_rdms,
        target_rdms=fmri_rdms,
        desc="Calculating Spearman partial correlations between RDMs",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
    )

    print("\nComputing finished!")

    return corrs


def rdms_corr_perm(model_rdms, eeg_rdms, n_permutation=1000, parallel=False,
                   n_jobs=-1, backend="loky", random_state=None):
    """
    Calculate permutation-based Spearman correlations between model RDMs
    and EEG-like RDMs.

    The permutation is performed by shuffling the condition labels of each
    EEG-like RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    eeg_rdms : array_like, shape [..., n_cons, n_cons]
        EEG-like RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    parallel : bool, default=False
        Whether to parallelize across EEG-like target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [..., n_permutation, n_models, 2]
        Permutation Spearman correlation results.
    """

    eeg_rdms = np.asarray(eeg_rdms)

    if eeg_rdms.ndim < 2 or eeg_rdms.ndim > 5:
        raise ValueError(
            "The shape of eeg_rdms should be [n_cons, n_cons], "
            "[n1, n_cons, n_cons], [n1, n2, n_cons, n_cons], or "
            "[n1, n2, n3, n_cons, n_cons]."
        )

    if eeg_rdms.shape[-1] != eeg_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of eeg_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing permutation-based Spearman correlations between model RDMs and EEG-like RDMs")

    perm_corrs = _compute_spearman_corrs_perm(
        model_rdms=model_rdms,
        target_rdms=eeg_rdms,
        n_permutation=n_permutation,
        desc="Calculating permutation-based Spearman correlations",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=random_state,
    )

    print("\nComputing finished!")

    return perm_corrs


def fmrirdms_corr_perm(model_rdms, fmri_rdms, n_permutation=1000, parallel=False,
                       n_jobs=-1, backend="loky", random_state=None):
    """
    Calculate permutation-based Spearman correlations between model RDMs
    and fMRI searchlight RDMs.

    The permutation is performed by shuffling the condition labels of each
    fMRI searchlight RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    fmri_rdms : array_like, shape [n_x, n_y, n_z, n_cons, n_cons]
        fMRI searchlight RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    parallel : bool, default=False
        Whether to parallelize across fMRI searchlight units.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [n_x, n_y, n_z, n_permutation, n_models, 2]
        Permutation Spearman correlation results.
    """

    fmri_rdms = np.asarray(fmri_rdms)

    if fmri_rdms.ndim != 5:
        raise ValueError(
            "The shape of fmri_rdms should be [n_x, n_y, n_z, n_cons, n_cons]."
        )

    if fmri_rdms.shape[-1] != fmri_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of fmri_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing permutation-based Spearman correlations between model RDMs and fMRI searchlight RDMs")

    perm_corrs = _compute_spearman_corrs_perm(
        model_rdms=model_rdms,
        target_rdms=fmri_rdms,
        n_permutation=n_permutation,
        desc="Calculating permutation-based Spearman correlations",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=random_state,
    )

    print("\nComputing finished!")

    return perm_corrs


def rdms_partialcorr_perm(model_rdms, eeg_rdms, n_permutation=1000, parallel=False,
                          n_jobs=-1, backend="loky", random_state=None):
    """
    Calculate permutation-based Spearman partial correlations between model RDMs
    and EEG-like RDMs.

    The permutation is performed by shuffling the condition labels of each
    EEG-like RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    eeg_rdms : array_like, shape [..., n_cons, n_cons]
        EEG-like RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    parallel : bool, default=False
        Whether to parallelize across EEG-like target RDMs.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [..., n_permutation, n_models, 2]
        Permutation Spearman partial correlation results.
    """

    eeg_rdms = np.asarray(eeg_rdms)

    if eeg_rdms.ndim < 2 or eeg_rdms.ndim > 5:
        raise ValueError(
            "The shape of eeg_rdms should be [n_cons, n_cons], "
            "[n1, n_cons, n_cons], [n1, n2, n_cons, n_cons], or "
            "[n1, n2, n3, n_cons, n_cons]."
        )

    if eeg_rdms.shape[-1] != eeg_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of eeg_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing permutation-based Spearman partial correlations between model RDMs and EEG-like RDMs")

    perm_corrs = _compute_partialcorrs_perm(
        model_rdms=model_rdms,
        target_rdms=eeg_rdms,
        n_permutation=n_permutation,
        desc="Calculating permutation-based Spearman partial correlations",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=random_state,
    )

    print("\nComputing finished!")

    return perm_corrs


def fmrirdms_partialcorr_perm(model_rdms, fmri_rdms, n_permutation=1000, parallel=False,
                              n_jobs=-1, backend="loky", random_state=None):
    """
    Calculate permutation-based Spearman partial correlations between model RDMs
    and fMRI searchlight RDMs.

    The permutation is performed by shuffling the condition labels of each
    fMRI searchlight RDM.

    Parameters
    ----------
    model_rdms : array_like, shape [n_models, n_cons, n_cons]
        Model RDMs.

    fmri_rdms : array_like, shape [n_x, n_y, n_z, n_cons, n_cons]
        fMRI searchlight RDMs.

    n_permutation : int, default=1000
        Number of condition-label permutations.

    parallel : bool, default=False
        Whether to parallelize across fMRI searchlight units.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"threading", "loky"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    perm_corrs : ndarray, shape [n_x, n_y, n_z, n_permutation, n_models, 2]
        Permutation Spearman partial correlation results.
    """

    fmri_rdms = np.asarray(fmri_rdms)

    if fmri_rdms.ndim != 5:
        raise ValueError(
            "The shape of fmri_rdms should be [n_x, n_y, n_z, n_cons, n_cons]."
        )

    if fmri_rdms.shape[-1] != fmri_rdms.shape[-2]:
        raise ValueError(
            "The last two dimensions of fmri_rdms should be square: [n_cons, n_cons]."
        )

    print("\nComputing permutation-based Spearman partial correlations between model RDMs and fMRI searchlight RDMs")

    perm_corrs = _compute_partialcorrs_perm(
        model_rdms=model_rdms,
        target_rdms=fmri_rdms,
        n_permutation=n_permutation,
        desc="Calculating permutation-based Spearman partial correlations",
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=random_state,
    )

    print("\nComputing finished!")

    return perm_corrs