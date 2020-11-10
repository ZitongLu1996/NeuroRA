# -*- coding: utf-8 -*-

' a module for testing neurora.corr_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.corr_cal import bhvANDeeg_corr, bhvANDfmri_corr, eegANDfmri_corr

class test_corr_cal(unittest.TestCase):

    def test_bhvANDeeg_corr(self):

        bhv_data = np.random.rand(10, 10, 20)
        eeg_data = np.random.rand(10, 10, 15, 32, 50)
        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=0, chl_opt=0, time_opt=0)
        self.assertEqual(corrs.shape[0], 2)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=0, chl_opt=0, time_opt=1)
        self.assertEqual(corrs.shape[0], 10)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=0, chl_opt=1, time_opt=0)
        self.assertEqual(corrs.shape[0], 32)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=0, chl_opt=1, time_opt=1)
        self.assertEqual(corrs.shape[0], 32)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=1, chl_opt=0, time_opt=0)
        self.assertEqual(corrs.shape[0], 10)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=1, chl_opt=0, time_opt=1)
        self.assertEqual(corrs.shape[0], 10)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=1, chl_opt=1, time_opt=0)
        self.assertEqual(corrs.shape[0], 10)

        corrs = bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=1, chl_opt=1, time_opt=1)
        self.assertEqual(corrs.shape[0], 10)

        bhv_data = np.random.rand(10, 8)
        output = bhvANDeeg_corr(bhv_data, eeg_data)
        self.assertEqual(output, "Invalid input!")

    def test_bhvANDfmri_corr(self):

        bhv_data = np.random.rand(8, 10, 20)
        fmri_data = np.random.rand(8, 10, 13, 23, 12)
        corrs = bhvANDfmri_corr(bhv_data, fmri_data, sub_result=0)
        self.assertEqual(corrs.shape[0], 11)

        corrs = bhvANDfmri_corr(bhv_data, fmri_data, sub_result=1)
        self.assertEqual(corrs.shape[0], 10)

        bhv_data = np.random.rand(10, 8)
        output = bhvANDeeg_corr(bhv_data, fmri_data)
        self.assertEqual(output, "Invalid input!")

    def test_eegANDfmri_corr(self):

        eeg_data = np.random.rand(8, 10, 15, 32, 50)
        fmri_data = np.random.rand(8, 10, 13, 5, 5)
        corrs = eegANDfmri_corr(eeg_data, fmri_data, chl_opt=0, sub_result=0)
        self.assertEqual(corrs.shape[0], 11)

        corrs = eegANDfmri_corr(eeg_data, fmri_data, chl_opt=0, sub_result=1)
        self.assertEqual(corrs.shape[0], 10)

        corrs = eegANDfmri_corr(eeg_data, fmri_data, chl_opt=1, sub_result=0)
        self.assertEqual(corrs.shape[0], 32)

        corrs = eegANDfmri_corr(eeg_data, fmri_data, chl_opt=1, sub_result=1)
        self.assertEqual(corrs.shape[0], 10)

        eeg_data = np.random.rand(8, 10, 15, 32)
        output = eegANDfmri_corr(eeg_data, fmri_data)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()