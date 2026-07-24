# -*- coding: utf-8 -*-

' a module for testing neurora.stps_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.stps_cal import stps, stps_fmri, stps_fmri_roi

class test_stps_cal(unittest.TestCase):

    def test_stps(self):

        data = np.random.rand(5, 20, 32, 25)
        label_item = np.zeros([20])
        label_item[:10] = 1
        label_rf = np.zeros([20])
        label_rf[:5] = 1
        label_rf[15:] = 1
        output = stps(data, label_item, label_rf)
        self.assertEqual(output.shape[0], 5)

        data = np.random.rand(20, 32, 50)
        output = stps(data, label_item, label_rf)
        self.assertEqual(output, "Invalid input!")

    def test_stps_fmri(self):

        fmri_data = np.random.rand(5, 20, 13, 24, 12)
        label_item = np.zeros([20])
        label_item[:10] = 1
        label_rf = np.zeros([20])
        label_rf[:5] = 1
        label_rf[15:] = 1
        output = stps_fmri(fmri_data, label_item, label_rf)
        self.assertEqual(output.shape[0], 5)

        fmri_data = np.random.rand(5, 20, 13, 24)
        output = stps_fmri(fmri_data, label_item, label_rf)
        self.assertEqual(output, "Invalid input!")

    def test_stps_fmri_roi(self):

        fmri_data = np.random.rand(5, 20, 13, 24, 12)
        label_item = np.zeros([20])
        label_item[:10] = 1
        label_rf = np.zeros([20])
        label_rf[:5] = 1
        label_rf[15:] = 1
        mask_data = np.random.randint(0, 2, (13, 24, 12))
        output = stps_fmri_roi(fmri_data, mask_data, label_item, label_rf)
        self.assertEqual(output.shape[0], 5)

        fmri_data = np.random.rand(5, 13, 24)
        output = stps_fmri_roi(fmri_data, mask_data, label_item, label_rf)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()