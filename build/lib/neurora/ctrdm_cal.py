# -*- coding: utf-8 -*-

' a module for calculating the cross-temporal RDM based on EEG-like data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr
from neurora.stuff import show_progressbar

'a function for calculating the cross-temporal RDM(s) based on EEG-like data'

def ctRDM(data, sub_opt=1, chl_opt=0, time_win=5, time_step=5):

    """
    Calculate CTRDMs for EEG-like data

    Parameters
    ----------
    data : array
        EEG/MEG data from a time-window.
        The shape of data must be [n_cons, n_subs, n_chls, n_ts]. n_cons, n_subs, n_chls & n_ts represent the number of
        conditions, the number of subjects, the number of channels and the number of time-points, respectively.
    sub_opt : int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    chl_opt : int 0 or 1. Default is 0.
        Caculate the CTRDMs for each channel or not.
        If chl_opt=1, calculate the CTRDMs for each channel.
        If chl_opt=0, calculate the CTRDMs after averaging the channels.
    time_win : int. Default is 5.
        Set a time-window for calculating the CTRDM for different time-points.
        If time_win=10, that means each calculation process based on 10 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.

    Returns
    -------
    CTRDMs : array
        Cross-Temporal RDMs.
        if chl_opt=1, the shape of CTRDMs is [n_subs, n_chls, int((n_ts-time_win)/time_step)+1,
        int((n_ts-time_win)/time_step)+1, n_cons, n_cons]
        if chl_opt=0, the shape of CTRDMs is [n_subs, int((n_ts-time_win)/time_step)+1,
        int((n_ts-time_win)/time_step)+1, n_cons, n_cons]
    """

    n_cons, n_subs, n_chls, n_ts = np.shape(data)

    nts = int((n_ts - time_win) / time_step) + 1

    data_for_cal = np.zeros([n_cons, n_subs, nts, n_chls, time_win])

    for con in range(n_cons):
        for sub in range(n_subs):
            for t in range(nts):
                for chl in range(n_chls):
                    data_for_cal[con, sub, t, chl] = data[con, sub, chl, t * time_step:t * time_step + time_win]

    # chl_opt=0
    if chl_opt == 0:

        data_for_cal = np.reshape(data_for_cal, [n_cons, n_subs, nts, n_chls*time_win])

        ctrdms = np.zeros([n_subs, nts, nts, n_cons, n_cons])

        total = n_subs * nts * nts

        for sub in range(n_subs):
            for t1 in range(nts):
                for t2 in range(nts):

                    # show the progressbar
                    percent = (sub * nts * nts + t1 * nts + t2 + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    for con1 in range(n_cons):
                        for con2 in range(n_cons):

                            if con1 != con2:
                                r = pearsonr(data_for_cal[con1, sub, t1], data_for_cal[con2, sub, t2])[0]
                                ctrdms[sub, t1, t2, con1, con2] = 1 - r
                            if con1 == con2:
                                ctrdms[sub, t1, t2, con1, con2] = 0

        # chl_opt=0 & sub_opt=0
        if sub_opt == 0:

            print("\nCross-temporal RDMs computing finished!")

            return np.average(ctrdms, axis=0)

        # chl_opt=0 & sub_opt=1
        else:

            print("\nCross-temporal RDMs computing finished!")

            return ctrdms

    # chl_opt=1
    else:

        ctrdms = np.zeros([n_subs, n_chls, nts, nts, n_cons, n_cons])

        total = n_subs * n_chls * nts * nts

        for sub in range(n_subs):
            for chl in range(n_chls):
                for t1 in range(nts):
                    for t2 in range(nts):

                        # show the progressbar
                        percent = (sub * nts * nts + t1 * nts + t2 + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        for con1 in range(n_cons):
                            for con2 in range(n_cons):

                                if con1 != con2:
                                    r = pearsonr(data_for_cal[con1, sub, t1, chl], data_for_cal[con2, sub, t2, chl])[0]
                                    ctrdms[sub, chl, t1, t2, con1, con2] = 1 - r
                                if con1 == con2:
                                    ctrdms[sub, chl, t1, t2, con1, con2] = 0

        # chl_opt=1 & sub_opt=0
        if sub_opt == 0:

            print("\nCross-temporal RDMs computing finished!")

            return np.average(ctrdms, axis=0)

        # chl_opt=1 & sub_opt=1
        else:

            print("\nCross-temporal RDMs computing finished!")

            return ctrdms