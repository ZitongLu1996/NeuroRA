# -*- coding: utf-8 -*-

' a module for testing neurora.nps_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.nps_cal import nps, nps_fmri, nps_fmri_roi

class test_nps_cal(unittest.TestCase):

    def test_nps(self):

        data = np.random.rand(2, 10, 15, 32, 50)
        output = nps(data, sub_opt=0)
        self.assertEqual(output.shape[0], 32)

        output = nps(data, sub_opt=1)
        self.assertEqual(output.shape[0], 10)

        data = np.random.rand(1, 10, 15, 32, 50)
        output = nps(data)
        self.assertEqual(output, "Invalid input!")

    def test_nps_fmri(self):

        fmri_data = np.random.rand(2, 10, 13, 23, 12)
        output = nps_fmri(fmri_data)
        self.assertEqual(output.shape[0], 10)

        fmri_data = np.random.rand(1, 10, 13, 23, 12)
        output = nps_fmri(fmri_data)
        self.assertEqual(output, "Invalid input!")

    def test_nps_fmri_roi(self):

        fmri_data = np.random.rand(2, 10, 13, 23, 12)
        mask_data = np.random.randint(0, 2, (13, 23, 12))
        output = nps_fmri_roi(fmri_data, mask_data)
        self.assertEqual(output.shape[0], 10)

        fmri_data = np.random.rand(1, 10, 13, 23, 12)
        output = nps_fmri_roi(fmri_data, mask_data)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()