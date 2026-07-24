# -*- coding: utf-8 -*-

' a module for testing neurora.corr_cal_by_rdm module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.corr_cal_by_rdm import rdms_corr, fmrirdms_corr

class test_corr_cal_by_rdm(unittest.TestCase):

    def test_rdms_corr(self):

        demo_rdm = np.random.rand(8, 8)
        eeg_rdms = np.random.rand(8, 8)
        output = rdms_corr(demo_rdm, eeg_rdms)
        self.assertEqual(output.shape[0], 2)

        eeg_rdms = np.random.rand(5, 8, 8)
        output = rdms_corr(demo_rdm, eeg_rdms)
        self.assertEqual(output.shape[0], 5)

        eeg_rdms = np.random.rand(4, 5, 8, 8)
        output = rdms_corr(demo_rdm, eeg_rdms)
        self.assertEqual(output.shape[0], 4)

        eeg_rdms = np.random.rand(3, 4, 5, 8, 8)
        output = rdms_corr(demo_rdm, eeg_rdms)
        self.assertEqual(output.shape[0], 3)

        eeg_rdms = np.random.rand(2, 3, 4, 5, 8, 8)
        output = rdms_corr(demo_rdm, eeg_rdms)
        self.assertEqual(output, "Invalid input!")

    def test_fmrirdms_corr(self):

        demo_rdm = np.random.rand(8, 8)
        fmri_rdms = np.random.rand(11, 23, 10, 8, 8)
        output = fmrirdms_corr(demo_rdm, fmri_rdms)
        self.assertEqual(output.shape[0], 11)

        fmri_rdms = np.random.rand(11, 23, 10, 8)
        output = fmrirdms_corr(demo_rdm, fmri_rdms)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()