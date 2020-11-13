# -*- coding: utf-8 -*-

' a module for testing neuora.rdm_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.rdm_cal import bhvRDM, eegRDM, fmriRDM, fmriRDM_roi

class test_rdm_cal(unittest.TestCase):

    """def test_bhvRDM(self):

        bhv_data = np.random.rand(8, 10, 20)
        rdm = bhvRDM(bhv_data, sub_opt=0)
        self.assertEqual(rdm.shape[0], 8)

        rdm = bhvRDM(bhv_data, sub_opt=1)
        self.assertEqual(rdm.shape[0], 10)

        bhv_data = np.random.rand(8, 10)
        output = bhvRDM(bhv_data)
        self.assertEqual(output, "Invalid input!")"""

    def test_eegRDM(self):

        eeg_data = np.random.rand(8, 10, 15, 32, 50)
        """rdms = eegRDM(eeg_data, sub_opt=0, chl_opt=0, time_opt=0)
        self.assertEqual(rdms.shape[0], 8)"""

        rdms = eegRDM(eeg_data, sub_opt=0, chl_opt=0, time_opt=1)
        self.assertEqual(rdms.shape[0], 8)

        rdms = eegRDM(eeg_data, sub_opt=0, chl_opt=1, time_opt=0)
        self.assertEqual(rdms.shape[0], 32)

        rdms = eegRDM(eeg_data, sub_opt=0, chl_opt=1, time_opt=1)
        self.assertEqual(rdms.shape[0], 32)

        rdms = eegRDM(eeg_data, sub_opt=1, chl_opt=0, time_opt=0)
        self.assertEqual(rdms.shape[0], 10)

        rdms = eegRDM(eeg_data, sub_opt=1, chl_opt=0, time_opt=1)
        self.assertEqual(rdms.shape[0], 10)

        rdms = eegRDM(eeg_data, sub_opt=1, chl_opt=1, time_opt=0)
        self.assertEqual(rdms.shape[0], 10)

        rdms = eegRDM(eeg_data, sub_opt=1, chl_opt=1, time_opt=1)
        self.assertEqual(rdms.shape[0], 10)

        eeg_data = np.random.rand(8, 10, 15, 32)
        output = bhvRDM(eeg_data)
        self.assertEqual(output, "Invalid input!")

    """def test_fmriRDM(self):

        fmri_data = np.random.rand(8, 10, 13, 23, 12)
        rdms = fmriRDM(fmri_data, sub_result=0)
        self.assertEqual(rdms.shape[0], 11)

        rdms = fmriRDM(fmri_data, sub_result=1)
        self.assertEqual(rdms.shape[0], 10)

    def test_fmriRDM_roi(self):

        fmri_data = np.random.rand(8, 10, 13, 23, 12)
        mask_data = np.random.randint(0, 2, (13, 23, 12))
        rdm = fmriRDM_roi(fmri_data, mask_data)
        self.assertEqual(rdm.shape[0], 8)"""

if __name__ == '__main__':
    unittest.main()
    

