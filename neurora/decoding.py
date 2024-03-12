# -*- coding: utf-8 -*-

' a module for classification-based neural decoding '

__author__ = 'Zitong Lu'

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from neurora.stuff import show_progressbar, smooth_1d, smooth_2d

np.seterr(divide='ignore', invalid='ignore')

' a function for time-by-time decoding for EEG-like data (cross validation) '

def tbyt_decoding_kfold(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, nfolds=5, nrepeats=2,
                        normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct time-by-time decoding for EEG-like data (cross validation)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    nfolds : int. Default is 5.
        The number of folds.
        k should be at least 2.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The time-by-time decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        x_train = xt[train_index]
                        x_test = xt[test_index]

                        if normalization is True:
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)

                        if pca is True:

                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)

                        svm = SVC(kernel='linear', tol=1e-4, probability=False)
                        svm.fit(x_train, y[train_index])
                        subacc[i, t, fold_index] = svm.score(x_test, y[test_index])

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 2))

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls*time_win, newnts])

        acc = np.zeros([nsubs, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        x_train = xt[train_index]
                        x_test = xt[test_index]

                        if normalization is True:
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)

                        if pca is True:

                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)

                        svm = SVC(kernel='linear', tol=1e-4, probability=False)
                        svm.fit(x_train, y[train_index])
                        subacc[i, t, fold_index] = svm.score(x_test, y[test_index])

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 2))

    if smooth is False:

        return acc

    if smooth is True:

        smooth_acc = smooth_1d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_1d(acc, n=smooth)

        return smooth_acc


' a function for time-by-time decoding for EEG-like data (hold out) '

def tbyt_decoding_holdout(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, iter=10,
                          test_size=0.3, normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct time-by-time decoding for EEG-like data (hold out)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    iter : int. Default is 10.
        The times for iteration.
    test_size : float. Default is 0.3.
        The proportion of the test set.
        test_size should be between 0.0 and 1.0.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The time-by-time decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts])

        total = nsubs * iter * newnts

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([iter, newnts])

            for i in range(iter):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    percent = (sub * iter * newnts + i * newnts + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    state = np.random.randint(0, 100)
                    xt = x[:, :, t]
                    x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                        random_state=state)

                    if normalization is True:

                        scaler = StandardScaler()
                        x_train = scaler.fit_transform(x_train)
                        x_test = scaler.transform(x_test)

                    if pca is True:
                        Pca = PCA(n_components=pca_components)
                        x_train = Pca.fit_transform(x_train)
                        x_test = Pca.transform(x_test)

                    svm = SVC(kernel='linear', tol=1e-4, probability=False)
                    svm.fit(x_train, y_train)
                    subacc[i, t] = svm.score(x_test, y_test)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls * time_win, newnts])

        acc = np.zeros([nsubs, newnts])

        total = nsubs * iter * newnts

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([iter, newnts])

            for i in range(iter):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    percent = (sub * iter * newnts + i * newnts + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    state = np.random.randint(0, 100)
                    xt = x[:, :, t]
                    x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                        random_state=state)

                    if normalization is True:
                        scaler = StandardScaler()
                        x_train = scaler.fit_transform(x_train)
                        x_test = scaler.transform(x_test)

                    if pca is True:
                        Pca = PCA(n_components=pca_components)
                        x_train = Pca.fit_transform(x_train)
                        x_test = Pca.transform(x_test)

                    svm = SVC(kernel='linear', tol=1e-4, probability=False)
                    svm.fit(x_train, y_train)
                    subacc[i, t] = svm.score(x_test, y_test)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_1d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_1d(acc, n=smooth)

        return smooth_acc


' a function for cross-temporal decoding for EEG-like data (cross validation) '

def ct_decoding_kfold(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, nfolds=5, nrepeats=2,
                      normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct cross-temporal decoding for EEG-like data (cross validation)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    nfolds : int. Default is 5.
        The number of folds.
        nfolds should be at least 2.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The cross-temporal decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if normalization is True:
                            if pca is True:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                x_test = scaler.transform(xt[test_index])
                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(x_train)
                                x_test = Pca.transform(x_test)
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])

                            if pca is False:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(scaler.transform(xt[test_index]), y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if normalization is False:
                            if pca is False:

                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(xt[train_index], y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(xt[test_index], y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(xtt[test_index], y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(xtt[test_index], y[test_index])

                            if pca is True:

                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(xt[train_index])
                                x_test = Pca.transform(xt[test_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(Pca.transform(xtt[test_index]), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(Pca.transform(xtt[test_index]), y[test_index])

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 3))

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls * time_win, newnts])

        acc = np.zeros([nsubs, newnts, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if normalization is True:
                            if pca is True:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                x_test = scaler.transform(xt[test_index])
                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(x_train)
                                x_test = Pca.transform(x_test)
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])

                            if pca is False:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(scaler.transform(xt[test_index]), y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if normalization is False:
                            if pca is False:

                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(xt[train_index], y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(xt[test_index], y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(xtt[test_index], y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(xtt[test_index], y[test_index])

                            if pca is True:

                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(xt[train_index])
                                x_test = Pca.transform(xt[test_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(Pca.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(Pca.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 3))

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_2d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_2d(acc, n=smooth)

        return smooth_acc


' a function for cross-temporal decoding for EEG-like data (hold-out) '

def ct_decoding_holdout(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, iter=10, test_size=0.3,
                        normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct cross-temporal decoding for EEG-like data (hold-out)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    iter : int. Default is 10.
        The times for iteration.
    test_size : float. Default is 0.3.
        The proportion of the test set.
        test_size should be between 0.0 and 1.0.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The cross-temporal decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts, newnts])

        total = nsubs * iter * newnts

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([iter, newnts, newnts])

            for i in range(iter):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    percent = (sub * iter * newnts + i * newnts + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    if normalization is True:
                        if pca is True:
                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)
                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(Pca.transform(scaler.transform(x_testt)), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(Pca.transform(scaler.transform(x_testt)), y_test)

                        if pca is False:
                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(scaler.transform(x_test), y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(scaler.transform(x_testt), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(scaler.transform(x_testt), y_test)

                    if normalization is False:
                        if pca is False:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(x_testt, y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(x_testt, y_test)

                        if pca is True:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(Pca.transform(x_testt), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(Pca.transform(x_testt), y_test)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls * time_win, newnts])

        acc = np.zeros([nsubs, newnts, newnts])

        total = nsubs * iter * newnts

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([iter, newnts, newnts])

            for i in range(iter):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    percent = (sub * iter * newnts + i * newnts + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    if normalization is True:
                        if pca is True:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)
                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(Pca.transform(scaler.transform(x_testt)), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(Pca.transform(scaler.transform(x_testt)), y_test)

                        if pca is False:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(scaler.transform(x_test), y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(scaler.transform(x_testt), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(scaler.transform(x_testt), y_test)

                    if normalization is False:
                        if pca is False:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(x_testt, y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(x_testt, y_test)

                        if pca is True:

                            state = np.random.randint(0, 100)
                            xt = x[:, :, t]
                            x_train, x_test, y_train, y_test = train_test_split(xt, y, test_size=test_size,
                                                                                random_state=state)
                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(x_train, y_train)
                            subacc[i, t, t] = svm.score(x_test, y_test)
                            for tt in range(newnts - 1):
                                if tt < t:
                                    xtt = x[:, :, tt]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt] = svm.score(Pca.transform(x_testt), y_test)
                                if tt >= t:
                                    xtt = x[:, :, tt + 1]
                                    x_train, x_testt, y_train, y_test = train_test_split(xtt, y, test_size=test_size,
                                                                                         random_state=state)
                                    subacc[i, t, tt + 1] = svm.score(Pca.transform(x_testt), y_test)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_2d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_2d(acc, n=smooth)

        return smooth_acc


' a function for unidirectional transfer decoding for EEG-like data '

def unidirectional_transfer_decoding(data1, labels1, data2, labels2, n=2, navg=5, time_opt="average", time_win=5,
                                     time_step=5, iter=10, normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct unidirectional transfer decoding for EEG-like data

    Parameters
    ----------
    data1 : array
        The neural data under condition1.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels1 : array
        The labels of each trial under condition1.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    data2 : array
        The neural data under condition2.
    labels2 : array
        The labels of each trial under condition2.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    iter : int. Default is 10.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The unidirectional transfer decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts1-time_win)/time_step)+1, int((n_ts2-time_win)/time_step)+1].
    """

    if np.shape(data1)[0] != np.shape(labels1)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data2)[0] != np.shape(labels2)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data1)[0] != np.shape(data2)[0]:

        print("\nThe number of subjects of data1 doesn't match the number of subjects of data2.\n")

        return "Invalid input!"

    if np.shape(data1)[2] != np.shape(data2)[2]:

        print("\nThe number of channels of data1 doesn't match the number of channels of data2.\n")

        return "Invalid input!"

    if np.shape(data1)[1] != np.shape(labels1)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    if np.shape(data2)[1] != np.shape(labels2)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials1, nchls, nts1 = np.shape(data1)
    nsubs, ntrials2, nchls, nts2 = np.shape(data2)

    ncategories1 = np.zeros([nsubs], dtype=int)

    labels1 = np.array(labels1)

    for sub in range(nsubs):

        sublabels1_set = set(labels1[sub].tolist())
        ncategories1[sub] = len(sublabels1_set)

    if len(set(ncategories1.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories1[0]:

        print("\nThe number of categories for decoding doesn't match ncategories1 (" + str(ncategories1) + ")!\n")

        return "Invalid input!"

    ncategories2 = np.zeros([nsubs], dtype=int)

    labels2 = np.array(labels2)

    for sub in range(nsubs):
        sublabels2_set = set(labels2[sub].tolist())
        ncategories2[sub] = len(sublabels2_set)

    if len(set(ncategories2.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories2[0]:

        print("\nThe number of categories for decoding doesn't match ncategories2 (" + str(ncategories2) + ")!\n")

        return "Invalid input!"

    if ncategories1[0] != ncategories2[0]:

        print("\nThe number of categories of data1 doesn't match the number of categories of data2.\n")

        return "Invalid input!"

    categories = list(sublabels1_set)

    newnts1 = int((nts1-time_win)/time_step)+1
    newnts2 = int((nts2-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data1 = np.zeros([nsubs, ntrials1, nchls, newnts1])
        avgt_data2 = np.zeros([nsubs, ntrials2, nchls, newnts2])

        for t in range(newnts1):
            avgt_data1[:, :, :, t] = np.average(data1[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        for t in range(newnts2):
            avgt_data2[:, :, :, t] = np.average(data2[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts1, newnts2])

        total = nsubs * iter * newnts1

        print("\nDecoding")

        for sub in range(nsubs):

            ns1 = np.zeros([n], dtype=int)

            for i in range(ntrials1):
                for j in range(n):
                    if labels1[sub, i] == categories[j]:
                        ns1[j] = ns1[j] + 1

            minn1 = int(np.min(ns1) / navg)

            ns2 = np.zeros([n], dtype=int)

            for i in range(ntrials2):
                for j in range(n):
                    if labels2[sub, i] == categories[j]:
                        ns2[j] = ns2[j] + 1

            minn2 = int(np.min(ns2) / navg)

            subacc = np.zeros([iter, newnts1, newnts2])

            for i in range(iter):

                datai1 = np.zeros([n, minn1 * navg, nchls, newnts1])
                datai2 = np.zeros([n, minn2 * navg, nchls, newnts2])
                labelsi1 = np.zeros([n, minn1], dtype=int)
                labelsi2 = np.zeros([n, minn2], dtype=int)

                for j in range(n):
                    labelsi1[j] = j
                    labelsi2[j] = j

                randomindex1 = np.random.permutation(np.array(range(ntrials1)))
                randomindex2 = np.random.permutation(np.array(range(ntrials2)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials1):
                    for k in range(n):

                        if labels1[sub, randomindex1[j]] == categories[k] and m[k] < minn1 * navg:
                            datai1[k, m[k]] = avgt_data1[sub, randomindex1[j]]
                            m[k] = m[k] + 1

                m = np.zeros([n], dtype=int)

                for j in range(ntrials2):
                    for k in range(n):

                        if labels2[sub, randomindex2[j]] == categories[k] and m[k] < minn2 * navg:
                            datai2[k, m[k]] = avgt_data2[sub, randomindex2[j]]
                            m[k] = m[k] + 1

                avg_datai1 = np.zeros([n, minn1, nchls, newnts1])
                avg_datai2 = np.zeros([n, minn2, nchls, newnts2])

                for j in range(minn1):
                    avg_datai1[:, j] = np.average(datai1[:, j * navg:j * navg + navg], axis=1)

                for j in range(minn2):
                    avg_datai2[:, j] = np.average(datai2[:, j * navg:j * navg + navg], axis=1)

                x1 = np.reshape(avg_datai1, [n * minn1, nchls, newnts1])
                x2 = np.reshape(avg_datai2, [n * minn2, nchls, newnts2])
                y1 = np.reshape(labelsi1, [n * minn1])
                y2 = np.reshape(labelsi2, [n * minn2])

                for t in range(newnts1):

                    percent = (sub * iter * newnts1 + i * newnts1 + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    if normalization is False:
                        if pca is False:

                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            xt1 = x1[:, :, t]
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(xt2, y2)

                        if pca is True:

                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            xt1 = x1[:, :, t]
                            Pca = PCA(n_components=pca_components)
                            xt1 = Pca.fit_transform(xt1)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(Pca.transform(xt2), y2)

                    if normalization is True:
                        if pca is True:

                            scaler = StandardScaler()
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            xt1 = scaler.fit_transform(x1[:, :, t])
                            Pca = PCA(n_components=pca_components)
                            xt1 = Pca.fit_transform(xt1)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(Pca.transform(scaler.transform(xt2)), y2)

                        if pca is False:

                            scaler = StandardScaler()
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            xt1 = scaler.fit_transform(x1[:, :, t])

                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(scaler.transform(xt2), y2)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts1 - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if time_opt == "features":

        avgt_data1 = np.zeros([nsubs, ntrials1, nchls, time_win, newnts1])
        avgt_data2 = np.zeros([nsubs, ntrials2, nchls, time_win, newnts2])

        for t in range(newnts1):
            avgt_data1[:, :, :, :, t] = data1[:, :, :, t * time_step:t * time_step + time_win]

        for t in range(newnts2):
            avgt_data2[:, :, :, :, t] = data2[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data1 = np.reshape(avgt_data1, [nsubs, ntrials1, nchls * time_win, newnts1])
        avgt_data2 = np.reshape(avgt_data2, [nsubs, ntrials2, nchls * time_win, newnts2])

        acc = np.zeros([nsubs, newnts1, newnts2])

        total = nsubs * iter * newnts1

        print("\nDecoding")

        for sub in range(nsubs):

            ns1 = np.zeros([n], dtype=int)

            for i in range(ntrials1):
                for j in range(n):
                    if labels1[sub, i] == categories[j]:
                        ns1[j] = ns1[j] + 1

            minn1 = int(np.min(ns1) / navg)

            ns2 = np.zeros([n], dtype=int)

            for i in range(ntrials2):
                for j in range(n):
                    if labels2[sub, i] == categories[j]:
                        ns2[j] = ns2[j] + 1

            minn2 = int(np.min(ns2) / navg)

            subacc = np.zeros([iter, newnts1, newnts2])

            for i in range(iter):

                datai1 = np.zeros([n, minn1 * navg, nchls * time_win, newnts1])
                datai2 = np.zeros([n, minn2 * navg, nchls * time_win, newnts2])
                labelsi1 = np.zeros([n, minn1], dtype=int)
                labelsi2 = np.zeros([n, minn2], dtype=int)

                for j in range(n):
                    labelsi1[j] = j
                    labelsi2[j] = j

                randomindex1 = np.random.permutation(np.array(range(ntrials1)))
                randomindex2 = np.random.permutation(np.array(range(ntrials2)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials1):
                    for k in range(n):

                        if labels1[sub, randomindex1[j]] == categories[k] and m[k] < minn1 * navg:
                            datai1[k, m[k]] = avgt_data1[sub, randomindex1[j]]
                            m[k] = m[k] + 1

                m = np.zeros([n], dtype=int)

                for j in range(ntrials2):
                    for k in range(n):

                        if labels2[sub, randomindex2[j]] == categories[k] and m[k] < minn2 * navg:
                            datai2[k, m[k]] = avgt_data2[sub, randomindex2[j]]
                            m[k] = m[k] + 1

                avg_datai1 = np.zeros([n, minn1, nchls * time_win, newnts1])
                avg_datai2 = np.zeros([n, minn2, nchls * time_win, newnts2])

                for j in range(minn1):
                    avg_datai1[:, j] = np.average(datai1[:, j * navg:j * navg + navg], axis=1)

                for j in range(minn2):
                    avg_datai2[:, j] = np.average(datai2[:, j * navg:j * navg + navg], axis=1)

                x1 = np.reshape(avg_datai1, [n * minn1, nchls * time_win, newnts1])
                x2 = np.reshape(avg_datai2, [n * minn2, nchls * time_win, newnts2])
                y1 = np.reshape(labelsi1, [n * minn1])
                y2 = np.reshape(labelsi2, [n * minn2])

                for t in range(newnts1):

                    percent = (sub * iter * newnts1 + i * newnts1 + t + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    if normalization is False:
                        if pca is False:

                            xt1 = x1[:, :, t]
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(xt2, y2)

                        if pca is True:

                            xt1 = x1[:, :, t]
                            Pca = PCA(n_components=pca_components)
                            xt1 = Pca.fit_transform(xt1)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(Pca.transform(xt2), y2)


                    if normalization is True:
                        if pca is True:

                            scaler = StandardScaler()
                            xt1 = scaler.fit_transform(x1[:, :, t])
                            Pca = PCA(n_components=pca_components)
                            xt1 = Pca.fit_transform(xt1)
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(Pca.transform(scaler.transform(xt2)), y2)

                        if pca is False:

                            scaler = StandardScaler()
                            xt1 = scaler.fit_transform(x1[:, :, t])
                            svm = SVC(kernel='linear', tol=1e-4, probability=False)
                            svm.fit(xt1, y1)
                            for tt in range(newnts2):
                                xt2 = x2[:, :, tt]
                                subacc[i, t, tt] = svm.score(scaler.transform(xt2), y2)

                    if sub == (nsubs - 1) and i == (iter - 1) and t == (newnts1 - 1):
                        print("\nDecoding finished!\n")

            acc[sub] = np.average(subacc, axis=0)

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_2d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_2d(acc, n=smooth)

        return smooth_acc


' a function for bidirectional transfer decoding for EEG-like data '

def bidirectional_transfer_decoding(data1, labels1, data2, labels2, n=2, navg=5, time_opt="average", time_win=5,
                                    time_step=5, iter=10, normalization=False, pca=False, pca_components=0.95, smooth=True):

    """
    Conduct bidirectional transfer decoding for EEG-like data

    Parameters
    ----------
    data1 : array
        The neural data under condition1.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels1 : array
        The labels of each trials under condition1.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    data2 : array
        The neural data under condition2.
    labels2 : array
        The labels of each trials under condition2.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    iter : int. Default is 10.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    Con1toCon2_accuracies : array
        The 1 transfer to 2 decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts1-time_win)/time_step)+1, int((n_ts2-time_win)/time_step)+1].
    Con2toCon1_accuracies : array
        The 2 transfer to 1 decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts2-time_win)/time_step)+1, int((n_ts1-time_win)/time_step)+1].
    """

    Con1toCon2_accuracies, Con2toCon1_accuracies = unidirectional_transfer_decoding(data1, labels1, data2,
                                    labels2, n=n, navg=navg, time_opt=time_opt, time_win=time_win, time_step=time_step,
                                    iter=iter, normalization=normalization, pca=pca, pca_components=pca_components,
                                    smooth=smooth), unidirectional_transfer_decoding(data2, labels2, data1, labels1,
                                    n=n, navg=navg, time_opt=time_opt, time_win=time_win, time_step=time_step,
                                    iter=iter, normalization=normalization, pca=pca, pca_components=pca_components,
                                    smooth=smooth)

    return Con1toCon2_accuracies, Con2toCon1_accuracies