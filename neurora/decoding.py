# -*- coding: utf-8 -*-

' a module for classification-based neural decoding '

__author__ = 'Zitong Lu'

import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from neurora.stuff import _resolve_n_jobs, _validate_backend

np.seterr(divide='ignore', invalid='ignore')


def _validate_eeg_data_labels(data, labels, data_name="data", labels_name="labels"):
    """
    Validate EEG-like data and labels.

    Parameters
    ----------
    data : array_like, shape [n_subs, n_trials, n_chls, n_ts]
        EEG-like data.

    labels : array_like, shape [n_subs, n_trials]
        Trial labels.

    data_name : str
        Name used in error messages.

    labels_name : str
        Name used in error messages.

    Returns
    -------
    data : ndarray
        Validated data.

    labels : ndarray
        Validated labels.
    """

    data = np.asarray(data)
    labels = np.asarray(labels)

    if data.ndim != 4:
        raise ValueError(
            f"{data_name} should have shape [n_subs, n_trials, n_chls, n_ts]."
        )

    if labels.ndim != 2:
        raise ValueError(
            f"{labels_name} should have shape [n_subs, n_trials]."
        )

    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"The number of subjects in {data_name} does not match {labels_name}."
        )

    if data.shape[1] != labels.shape[1]:
        raise ValueError(
            f"The number of trials in {data_name} does not match {labels_name}."
        )

    return data, labels
    

def _get_categories(labels, n):
    """
    Validate labels and return category labels.

    Parameters
    ----------
    labels : ndarray, shape [n_subs, n_trials]
        Labels.

    n : int
        Expected number of categories.

    Returns
    -------
    categories : ndarray, shape [n]
        Category labels.
    """

    categories_per_sub = [np.unique(labels[sub]) for sub in range(labels.shape[0])]
    n_categories = np.array([len(cats) for cats in categories_per_sub], dtype=int)

    if len(set(n_categories.tolist())) != 1:
        raise ValueError(
            "Invalid labels: different subjects have different numbers of categories."
        )

    if n_categories[0] != n:
        raise ValueError(
            f"The number of categories in labels is {n_categories[0]}, "
            f"but n={n} was provided."
        )

    reference_categories = categories_per_sub[0]

    for sub, cats in enumerate(categories_per_sub[1:], start=1):
        if not np.array_equal(np.sort(cats), np.sort(reference_categories)):
            raise ValueError(
                "Invalid labels: category identities differ across subjects."
            )

    return reference_categories


def _make_subject_seeds(n_subjects, random_state=None):
    """
    Create one reproducible random seed per subject.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    seeds : ndarray, shape [n_subjects]
        Subject-level random seeds.
    """

    rng = np.random.default_rng(random_state)
    return rng.integers(0, np.iinfo(np.int32).max, size=n_subjects)


def _make_time_window_features(data, time_opt, time_win, time_step):
    """
    Convert EEG-like data into time-window features.

    Parameters
    ----------
    data : ndarray, shape [n_subs, n_trials, n_chls, n_ts]
        EEG-like neural data.

    time_opt : {"average", "features"}
        Strategy for handling time points within each time window.

        - "average": average time points within each window.
        - "features": use all time points within each window as features.

    time_win : int
        Size of the time window in samples.

    time_step : int
        Step size between adjacent time windows in samples.

    Returns
    -------
    windowed_data : ndarray, shape [n_subs, n_trials, n_features, n_windows]
        Time-windowed data.

        If time_opt="average", n_features = n_chls.
        If time_opt="features", n_features = n_chls * time_win.
    """

    data = np.asarray(data)

    if time_opt not in {"average", "features"}:
        raise ValueError("time_opt should be 'average' or 'features'.")

    if not isinstance(time_win, int) or time_win < 1:
        raise ValueError("time_win should be a positive integer.")

    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step should be a positive integer.")

    if data.ndim != 4:
        raise ValueError(
            "data should have shape [n_subs, n_trials, n_chls, n_ts]."
        )

    n_subs, n_trials, n_chls, n_ts = data.shape

    if time_win > n_ts:
        raise ValueError(
            "time_win should not be larger than the number of time points."
        )

    n_windows = int((n_ts - time_win) / time_step) + 1

    if time_opt == "average":
        windowed_data = np.empty(
            (n_subs, n_trials, n_chls, n_windows),
            dtype=data.dtype,
        )

        for t in range(n_windows):
            start = t * time_step
            stop = start + time_win
            windowed_data[..., t] = np.mean(data[..., start:stop], axis=3)

        return windowed_data

    windowed_data = np.empty(
        (n_subs, n_trials, n_chls * time_win, n_windows),
        dtype=data.dtype,
    )

    for t in range(n_windows):
        start = t * time_step
        stop = start + time_win
        window = data[..., start:stop]
        windowed_data[..., t] = window.reshape(
            n_subs,
            n_trials,
            n_chls * time_win,
        )

    return windowed_data


def _make_balanced_averaged_data(sub_data, sub_labels, categories, navg, rng):
    """
    Create balanced pseudo-trials by averaging navg trials within each category.

    Parameters
    ----------
    sub_data : ndarray, shape [n_trials, n_features, n_windows]
        Data from one subject.

    sub_labels : ndarray, shape [n_trials]
        Labels from one subject.

    categories : array_like, shape [n_categories]
        Category labels.

    navg : int
        Number of trials averaged into one pseudo-trial.

    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    x : ndarray, shape [n_categories * n_pseudotrials, n_features, n_windows]
        Balanced pseudo-trial data.

    y : ndarray, shape [n_categories * n_pseudotrials]
        Integer labels from 0 to n_categories-1.
    """

    if not isinstance(navg, int) or navg < 1:
        raise ValueError("navg should be a positive integer.")

    n_categories = len(categories)
    n_features = sub_data.shape[1]
    n_windows = sub_data.shape[2]

    category_indices = []
    n_available = []

    for cat in categories:
        idx = np.where(sub_labels == cat)[0]
        category_indices.append(idx)
        n_available.append(len(idx))

    n_pseudotrials = min(n_available) // navg

    if n_pseudotrials < 1:
        raise ValueError(
            "Not enough trials per category to create pseudo-trials. "
            "Try reducing navg."
        )

    x = np.empty(
        (n_categories, n_pseudotrials, n_features, n_windows),
        dtype=sub_data.dtype,
    )

    for ci, idx in enumerate(category_indices):
        selected = rng.permutation(idx)[: n_pseudotrials * navg]
        selected_data = sub_data[selected]
        selected_data = selected_data.reshape(
            n_pseudotrials,
            navg,
            n_features,
            n_windows,
        )
        x[ci] = selected_data.mean(axis=1)

    y = np.repeat(np.arange(n_categories), n_pseudotrials)
    x = x.reshape(n_categories * n_pseudotrials, n_features, n_windows)

    return x, y


def _make_repeated_balanced_data(sub_data, sub_labels, categories, navg,
                                 n_repeats, seed):
    """
    Create repeated balanced pseudo-trial datasets for one subject.

    Parameters
    ----------
    sub_data : ndarray, shape [n_trials, n_features, n_windows]
        Time-windowed data from one subject.

    sub_labels : ndarray, shape [n_trials]
        Labels from one subject.

    categories : array_like, shape [n_categories]
        Category labels.

    navg : int
        Number of trials averaged into one pseudo-trial.

    n_repeats : int
        Number of repeated random pseudo-trial averaging iterations.

    seed : int
        Random seed.

    Returns
    -------
    repeated_data : list of tuple
        Each element is (x, y), where:

        - x has shape [n_samples, n_features, n_windows]
        - y has shape [n_samples]
    """

    rng = np.random.default_rng(seed)
    repeated_data = []

    for _ in range(n_repeats):
        x, y = _make_balanced_averaged_data(
            sub_data=sub_data,
            sub_labels=sub_labels,
            categories=categories,
            navg=navg,
            rng=rng,
        )
        repeated_data.append((x, y))

    return repeated_data


def _make_repeated_transfer_data(sub_data_train, sub_labels_train, sub_data_test, sub_labels_test,
                                 categories, navg, n_iter, seed):
    """
    Create repeated balanced pseudo-trial datasets for transfer decoding.

    Parameters
    ----------
    sub_data_train : ndarray, shape [n_trials_train, n_features, n_train_windows]
        Training-condition data from one subject.

    sub_labels_train : ndarray, shape [n_trials_train]
        Training-condition labels from one subject.

    sub_data_test : ndarray, shape [n_trials_test, n_features, n_test_windows]
        Testing-condition data from one subject.

    sub_labels_test : ndarray, shape [n_trials_test]
        Testing-condition labels from one subject.

    categories : array_like, shape [n_categories]
        Category labels.

    navg : int
        Number of trials averaged into one pseudo-trial.

    n_iter : int
        Number of repeated pseudo-trial averaging iterations.

    seed : int
        Random seed.

    Returns
    -------
    repeated_data : list of tuple
        Each element is (x_train, y_train, x_test, y_test).
    """

    rng = np.random.default_rng(seed)
    repeated_data = []

    for _ in range(n_iter):
        x_train, y_train = _make_balanced_averaged_data(
            sub_data=sub_data_train,
            sub_labels=sub_labels_train,
            categories=categories,
            navg=navg,
            rng=rng,
        )

        x_test, y_test = _make_balanced_averaged_data(
            sub_data=sub_data_test,
            sub_labels=sub_labels_test,
            categories=categories,
            navg=navg,
            rng=rng,
        )

        repeated_data.append((x_train, y_train, x_test, y_test))

    return repeated_data


def _fit_preprocess_classifier(x_train, y_train, normalization=False, pca=False, pca_components=0.95):
    """
    Fit optional preprocessing and a linear SVM classifier.

    Parameters
    ----------
    x_train : ndarray, shape [n_samples, n_features]
        Training data.

    y_train : ndarray, shape [n_samples]
        Training labels.

    normalization : bool
        Whether to standardize features.

    pca : bool
        Whether to apply PCA.

    pca_components : int or float
        Number of PCA components or explained variance ratio.

    Returns
    -------
    svm : SVC
        Fitted linear SVM.

    scaler : StandardScaler or None
        Fitted scaler.

    pca_model : PCA or None
        Fitted PCA model.
    """

    scaler = None
    pca_model = None

    x_train_transformed = x_train

    if normalization:
        scaler = StandardScaler()
        x_train_transformed = scaler.fit_transform(x_train_transformed)

    if pca:
        pca_model = PCA(n_components=pca_components)
        x_train_transformed = pca_model.fit_transform(x_train_transformed)

    svm = SVC(kernel="linear", tol=1e-4, probability=False)
    svm.fit(x_train_transformed, y_train)

    return svm, scaler, pca_model


def _apply_preprocess(x, scaler=None, pca_model=None):
    """
    Apply fitted preprocessing to data.

    Parameters
    ----------
    x : ndarray, shape [n_samples, n_features]
        Input data.

    scaler : StandardScaler or None
        Fitted scaler.

    pca_model : PCA or None
        Fitted PCA model.

    Returns
    -------
    x : ndarray
        Transformed data.
    """

    if scaler is not None:
        x = scaler.transform(x)

    if pca_model is not None:
        x = pca_model.transform(x)

    return x


def _decode_tbyt_one_subject_time(repeated_data, time_idx, nfolds, normalization,
                                  pca, pca_components, seed):
    """
    Decode one subject at one time window.

    Parameters
    ----------
    repeated_data : list of tuple
        Repeated pseudo-trial datasets for one subject.

    time_idx : int
        Time-window index.

    nfolds : int
        Number of cross-validation folds.

    normalization : bool
        Whether to standardize features.

    pca : bool
        Whether to apply PCA.

    pca_components : int or float
        Number of PCA components or explained variance ratio.

    seed : int
        Random seed.

    Returns
    -------
    acc : float
        Averaged decoding accuracy for this subject and time window.
    """

    rng = np.random.default_rng(seed)
    fold_accs = []

    for x, y in repeated_data:
        xt = x[:, :, time_idx]

        kf_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        kf = StratifiedKFold(
            n_splits=nfolds,
            shuffle=True,
            random_state=kf_seed,
        )

        for train_idx, test_idx in kf.split(xt, y):
            svm, scaler, pca_model = _fit_preprocess_classifier(
                xt[train_idx],
                y[train_idx],
                normalization=normalization,
                pca=pca,
                pca_components=pca_components,
            )

            x_test = _apply_preprocess(
                xt[test_idx],
                scaler=scaler,
                pca_model=pca_model,
            )

            fold_accs.append(svm.score(x_test, y[test_idx]))

    return float(np.mean(fold_accs))


def _decode_ct_one_subject_train_time(repeated_data, train_time_idx, nfolds, normalization,
                                      pca, pca_components, seed):
    """
    Cross-temporal decoding for one subject and one training time window.

    Parameters
    ----------
    repeated_data : list of tuple
        Repeated pseudo-trial datasets for one subject.

    train_time_idx : int
        Training time-window index.

    nfolds : int
        Number of cross-validation folds.

    normalization : bool
        Whether to standardize features.

    pca : bool
        Whether to apply PCA.

    pca_components : int or float
        Number of PCA components or explained variance ratio.

    seed : int
        Random seed.

    Returns
    -------
    acc_vector : ndarray, shape [n_windows]
        Decoding accuracies across all testing time windows.
    """

    rng = np.random.default_rng(seed)
    n_windows = repeated_data[0][0].shape[2]
    accs = np.empty((len(repeated_data), n_windows, nfolds), dtype=float)

    for rep_idx, (x, y) in enumerate(repeated_data):
        x_train_time = x[:, :, train_time_idx]

        kf_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        kf = StratifiedKFold(
            n_splits=nfolds,
            shuffle=True,
            random_state=kf_seed,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_train_time, y)):
            svm, scaler, pca_model = _fit_preprocess_classifier(
                x_train_time[train_idx],
                y[train_idx],
                normalization=normalization,
                pca=pca,
                pca_components=pca_components,
            )

            for test_time_idx in range(n_windows):
                x_test_time = x[:, :, test_time_idx]
                x_test = _apply_preprocess(
                    x_test_time[test_idx],
                    scaler=scaler,
                    pca_model=pca_model,
                )

                accs[rep_idx, test_time_idx, fold_idx] = svm.score(
                    x_test,
                    y[test_idx],
                )

    return accs.mean(axis=(0, 2))


def _decode_transfer_one_subject_train_time(repeated_data, train_time_idx, normalization,
                                            pca, pca_components):
    """
    Transfer decoding for one subject and one training time window.

    Parameters
    ----------
    repeated_data : list of tuple
        Repeated transfer pseudo-trial datasets for one subject.

    train_time_idx : int
        Training time-window index.

    normalization : bool
        Whether to standardize features.

    pca : bool
        Whether to apply PCA.

    pca_components : int or float
        Number of PCA components or explained variance ratio.

    Returns
    -------
    acc_vector : ndarray, shape [n_test_windows]
        Transfer decoding accuracies across all testing time windows.
    """

    n_test_windows = repeated_data[0][2].shape[2]
    accs = np.empty((len(repeated_data), n_test_windows), dtype=float)

    for rep_idx, (x_train_all, y_train, x_test_all, y_test) in enumerate(repeated_data):
        svm, scaler, pca_model = _fit_preprocess_classifier(
            x_train_all[:, :, train_time_idx],
            y_train,
            normalization=normalization,
            pca=pca,
            pca_components=pca_components,
        )

        for test_time_idx in range(n_test_windows):
            x_test = _apply_preprocess(
                x_test_all[:, :, test_time_idx],
                scaler=scaler,
                pca_model=pca_model,
            )

            accs[rep_idx, test_time_idx] = svm.score(x_test, y_test)

    return accs.mean(axis=0)


def tbyt_decoding_kfold(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5,
                        nfolds=5, nrepeats=2, normalization=False, pca=False, pca_components=0.95,
                        parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Conduct time-by-time decoding for EEG-like data using cross-validation.

    Parameters
    ----------
    data : array_like, shape [n_subs, n_trials, n_chls, n_ts]
        EEG-like neural data.

    labels : array_like, shape [n_subs, n_trials]
        Trial labels for each subject.

    n : int, default=2
        Number of categories for classification.

    navg : int, default=5
        Number of trials averaged into each pseudo-trial.

    time_opt : {"average", "features"}, default="average"
        Strategy for handling time points within each time window.

        - "average": average time points within each window.
        - "features": use all time points within each window as features.

    time_win : int, default=5
        Size of the time window in samples.

    time_step : int, default=5
        Step size between adjacent time windows in samples.

    nfolds : int, default=5
        Number of cross-validation folds.

    nrepeats : int, default=2
        Number of repeated random pseudo-trial averaging iterations.

    normalization : bool, default=False
        Whether to standardize features using the training data.

    pca : bool, default=False
        Whether to apply PCA after optional normalization.

    pca_components : int or float, default=0.95
        Number of PCA components, or explained variance ratio.

    parallel : bool, default=False
        Whether to parallelize across subjects using joblib.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed for reproducible pseudo-trial averaging and cross-validation.

    Returns
    -------
    accuracies : ndarray, shape [n_subs, n_windows]
        Time-by-time decoding accuracies.
    """

    data, labels = _validate_eeg_data_labels(data, labels)
    categories = _get_categories(labels, n)

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if nfolds < 2:
        raise ValueError("nfolds should be at least 2.")

    windowed_data = _make_time_window_features(
        data=data,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
    )

    n_subs = windowed_data.shape[0]
    n_windows = windowed_data.shape[3]
    seeds = _make_subject_seeds(n_subs, random_state=random_state)

    print("\nTime-by-time decoding")

    subject_repeated_data = [
        _make_repeated_balanced_data(
            sub_data=windowed_data[sub],
            sub_labels=labels[sub],
            categories=categories,
            navg=navg,
            n_repeats=nrepeats,
            seed=int(seeds[sub]),
        )
        for sub in tqdm(range(n_subs), desc="Preparing pseudo-trials")
    ]

    acc = np.empty((n_subs, n_windows), dtype=float)

    tasks = [
        (sub, time_idx)
        for sub in range(n_subs)
        for time_idx in range(n_windows)
    ]

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_decode_tbyt_one_subject_time)(
                subject_repeated_data[sub],
                time_idx,
                nfolds,
                normalization,
                pca,
                pca_components,
                int(seeds[sub] + time_idx + 1),
            )
            for sub, time_idx in tqdm(tasks, desc="EEG Decoding")
        )

        for (sub, time_idx), value in zip(tasks, results):
            acc[sub, time_idx] = value

    else:
        for sub, time_idx in tqdm(tasks, desc="EEG Decoding"):
            acc[sub, time_idx] = _decode_tbyt_one_subject_time(
                subject_repeated_data[sub],
                time_idx,
                nfolds,
                normalization,
                pca,
                pca_components,
                int(seeds[sub] + time_idx + 1),
            )

    print("\nDecoding finished!\n")

    return acc


def ct_decoding_kfold(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5,
                      nfolds=5, nrepeats=2, normalization=False, pca=False, pca_components=0.95,
                      parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Conduct cross-temporal decoding for EEG-like data using cross-validation.

    Parameters
    ----------
    data : array_like, shape [n_subs, n_trials, n_chls, n_ts]
        EEG-like neural data.

    labels : array_like, shape [n_subs, n_trials]
        Trial labels for each subject.

    n : int, default=2
        Number of categories for classification.

    navg : int, default=5
        Number of trials averaged into each pseudo-trial.

    time_opt : {"average", "features"}, default="average"
        Strategy for handling time points within each time window.

    time_win : int, default=5
        Size of the time window in samples.

    time_step : int, default=5
        Step size between adjacent time windows in samples.

    nfolds : int, default=5
        Number of cross-validation folds.

    nrepeats : int, default=2
        Number of repeated random pseudo-trial averaging iterations.

    normalization : bool, default=False
        Whether to standardize features using the training data.

    pca : bool, default=False
        Whether to apply PCA after optional normalization.

    pca_components : int or float, default=0.95
        Number of PCA components, or explained variance ratio.

    parallel : bool, default=False
        Whether to parallelize across subjects using joblib.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed for reproducible pseudo-trial averaging and cross-validation.

    Returns
    -------
    accuracies : ndarray, shape [n_subs, n_windows, n_windows]
        Cross-temporal decoding accuracies.

        The second dimension is training time, and the third dimension is
        testing time.
    """

    data, labels = _validate_eeg_data_labels(data, labels)
    categories = _get_categories(labels, n)

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    if nfolds < 2:
        raise ValueError("nfolds should be at least 2.")

    windowed_data = _make_time_window_features(
        data=data,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
    )

    n_subs = windowed_data.shape[0]
    n_windows = windowed_data.shape[3]
    seeds = _make_subject_seeds(n_subs, random_state=random_state)

    print("\nCross-temporal decoding")

    subject_repeated_data = [
        _make_repeated_balanced_data(
            sub_data=windowed_data[sub],
            sub_labels=labels[sub],
            categories=categories,
            navg=navg,
            n_repeats=nrepeats,
            seed=int(seeds[sub]),
        )
        for sub in tqdm(range(n_subs), desc="Preparing pseudo-trials")
    ]

    acc = np.empty((n_subs, n_windows, n_windows), dtype=float)

    tasks = [
        (sub, train_time_idx)
        for sub in range(n_subs)
        for train_time_idx in range(n_windows)
    ]

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_decode_ct_one_subject_train_time)(
                subject_repeated_data[sub],
                train_time_idx,
                nfolds,
                normalization,
                pca,
                pca_components,
                int(seeds[sub] + train_time_idx + 1),
            )
            for sub, train_time_idx in tqdm(
                tasks,
                desc="EEG Decoding",
            )
        )

        for (sub, train_time_idx), acc_vector in zip(tasks, results):
            acc[sub, train_time_idx, :] = acc_vector

    else:
        for sub, train_time_idx in tqdm(tasks, desc="EEG Decoding"):
            acc[sub, train_time_idx, :] = _decode_ct_one_subject_train_time(
                subject_repeated_data[sub],
                train_time_idx,
                nfolds,
                normalization,
                pca,
                pca_components,
                int(seeds[sub] + train_time_idx + 1),
            )

    print("\nDecoding finished!\n")

    return acc


def unidirectional_transfer_decoding(data1, labels1, data2, labels2, n=2, navg=5, time_opt="average",
                                     time_win=5, time_step=5, n_iter=10, normalization=False,
                                     pca=False, pca_components=0.95,
                                     parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Conduct unidirectional transfer decoding for EEG-like data.

    The classifier is trained on data1 and tested on data2.

    Parameters
    ----------
    data1 : array_like, shape [n_subs, n_trials1, n_chls, n_ts1]
        Training-condition EEG-like data.

    labels1 : array_like, shape [n_subs, n_trials1]
        Labels for data1.

    data2 : array_like, shape [n_subs, n_trials2, n_chls, n_ts2]
        Testing-condition EEG-like data.

    labels2 : array_like, shape [n_subs, n_trials2]
        Labels for data2.

    n : int, default=2
        Number of categories for classification.

    navg : int, default=5
        Number of trials averaged into each pseudo-trial.

    time_opt : {"average", "features"}, default="average"
        Strategy for handling time points within each time window.

    time_win : int, default=5
        Size of the time window in samples.

    time_step : int, default=5
        Step size between adjacent time windows in samples.

    n_iter : int, default=10
        Number of repeated random pseudo-trial averaging iterations.

    normalization : bool, default=False
        Whether to standardize features using data1.

    pca : bool, default=False
        Whether to apply PCA after optional normalization.

    pca_components : int or float, default=0.95
        Number of PCA components, or explained variance ratio.

    parallel : bool, default=False
        Whether to parallelize across subjects using joblib.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.
        If n_jobs=-1, all available CPU cores will be used.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed for reproducible pseudo-trial averaging.

    Returns
    -------
    accuracies : ndarray, shape [n_subs, n_train_windows, n_test_windows]
        Unidirectional transfer decoding accuracies.
    """

    data1, labels1 = _validate_eeg_data_labels(data1, labels1, "data1", "labels1")
    data2, labels2 = _validate_eeg_data_labels(data2, labels2, "data2", "labels2")

    if data1.shape[0] != data2.shape[0]:
        raise ValueError("data1 and data2 should have the same number of subjects.")

    if data1.shape[2] != data2.shape[2]:
        raise ValueError("data1 and data2 should have the same number of channels.")

    categories1 = _get_categories(labels1, n)
    categories2 = _get_categories(labels2, n)

    if not np.array_equal(np.sort(categories1), np.sort(categories2)):
        raise ValueError("data1 and data2 should have the same category labels.")

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    windowed_data1 = _make_time_window_features(
        data=data1,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
    )

    windowed_data2 = _make_time_window_features(
        data=data2,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
    )

    n_subs = windowed_data1.shape[0]
    n_train_windows = windowed_data1.shape[3]
    n_test_windows = windowed_data2.shape[3]
    seeds = _make_subject_seeds(n_subs, random_state=random_state)

    print("\nUnidirectional transfer decoding")

    subject_repeated_data = [
        _make_repeated_transfer_data(
            sub_data_train=windowed_data1[sub],
            sub_labels_train=labels1[sub],
            sub_data_test=windowed_data2[sub],
            sub_labels_test=labels2[sub],
            categories=categories1,
            navg=navg,
            n_iter=n_iter,
            seed=int(seeds[sub]),
        )
        for sub in tqdm(range(n_subs), desc="Preparing pseudo-trials")
    ]

    acc = np.empty((n_subs, n_train_windows, n_test_windows), dtype=float)

    tasks = [
        (sub, train_time_idx)
        for sub in range(n_subs)
        for train_time_idx in range(n_train_windows)
    ]

    if parallel:
        n_jobs = _resolve_n_jobs(n_jobs)
        backend = _validate_backend(backend)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_decode_transfer_one_subject_train_time)(
                subject_repeated_data[sub],
                train_time_idx,
                normalization,
                pca,
                pca_components,
            )
            for sub, train_time_idx in tqdm(
                tasks,
                desc="EEG Decoding",
            )
        )

        for (sub, train_time_idx), acc_vector in zip(tasks, results):
            acc[sub, train_time_idx, :] = acc_vector

    else:
        for sub, train_time_idx in tqdm(tasks, desc="EEG Decoding"):
            acc[sub, train_time_idx, :] = _decode_transfer_one_subject_train_time(
                subject_repeated_data[sub],
                train_time_idx,
                normalization,
                pca,
                pca_components,
            )

    print("\nDecoding finished!\n")

    return acc


def bidirectional_transfer_decoding(data1, labels1, data2, labels2, n=2, navg=5, time_opt="average",
                                    time_win=5, time_step=5, n_iter=10, normalization=False,
                                    pca=False, pca_components=0.95,
                                    parallel=False, n_jobs=-1, backend="loky", random_state=None):
    """
    Conduct bidirectional transfer decoding for EEG-like data.

    Parameters
    ----------
    data1 : array_like, shape [n_subs, n_trials1, n_chls, n_ts1]
        Data from condition 1.

    labels1 : array_like, shape [n_subs, n_trials1]
        Labels for condition 1.

    data2 : array_like, shape [n_subs, n_trials2, n_chls, n_ts2]
        Data from condition 2.

    labels2 : array_like, shape [n_subs, n_trials2]
        Labels for condition 2.

    n : int, default=2
        Number of categories for classification.

    navg : int, default=5
        Number of trials averaged into each pseudo-trial.

    time_opt : {"average", "features"}, default="average"
        Strategy for handling time points within each time window.

    time_win : int, default=5
        Size of the time window in samples.

    time_step : int, default=5
        Step size between adjacent time windows in samples.

    n_iter : int, default=10
        Number of repeated random pseudo-trial averaging iterations.

    normalization : bool, default=False
        Whether to standardize features.

    pca : bool, default=False
        Whether to apply PCA.

    pca_components : int or float, default=0.95
        Number of PCA components, or explained variance ratio.

    parallel : bool, default=False
        Whether to parallelize across subjects using joblib.

    n_jobs : int, default=-1
        Number of workers used when parallel=True.

    backend : {"loky", "threading"}, default="loky"
        Joblib backend used when parallel=True.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    con1_to_con2_accuracies : ndarray, shape [n_subs, n_train_windows1, n_test_windows2]
        Transfer decoding accuracies from condition 1 to condition 2.

    con2_to_con1_accuracies : ndarray, shape [n_subs, n_train_windows2, n_test_windows1]
        Transfer decoding accuracies from condition 2 to condition 1.
    """

    if random_state is None:
        seed_1 = None
        seed_2 = None
    else:
        seed_1 = random_state
        seed_2 = random_state + 1

    if not isinstance(parallel, bool):
        raise TypeError("parallel should be True or False.")

    con1_to_con2_accuracies = unidirectional_transfer_decoding(
        data1,
        labels1,
        data2,
        labels2,
        n=n,
        navg=navg,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
        n_iter=n_iter,
        normalization=normalization,
        pca=pca,
        pca_components=pca_components,
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=seed_1,
    )

    con2_to_con1_accuracies = unidirectional_transfer_decoding(
        data2,
        labels2,
        data1,
        labels1,
        n=n,
        navg=navg,
        time_opt=time_opt,
        time_win=time_win,
        time_step=time_step,
        n_iter=n_iter,
        normalization=normalization,
        pca=pca,
        pca_components=pca_components,
        parallel=parallel,
        n_jobs=n_jobs,
        backend=backend,
        random_state=seed_2,
    )

    return con1_to_con2_accuracies, con2_to_con1_accuracies
