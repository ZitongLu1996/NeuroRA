# -*- coding: utf-8 -*-

' a module for testing neurora.isc_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.isc_cal import isc, isc_fmri, isc_fmri_roi

class test_isc_cal(unittest.TestCase):

    def test_isc(self):

        data = np.random.rand(5, 32, 25)
        output = isc(data)
        self.assertEqual(output.shape[0], 10)

        data = np.random.rand(5, 32)
        output = isc(data)
        self.assertEqual(output, "Invalid input!")

    def test_isc_fmri(self):

        fmri_data = np.random.rand(8, 5, 13, 24, 12)
        output = isc_fmri(fmri_data)
        self.assertEqual(output.shape[0], 8)

        fmri_data = np.random.rand(5, 20, 13, 24)
        output = isc_fmri(fmri_data)
        self.assertEqual(output, "Invalid input!")

    def test_isc_fmri_roi(self):

        fmri_data = np.random.rand(8, 5, 13, 24, 12)
        mask_data = np.random.randint(0, 2, (13, 24, 12))
        output = isc_fmri_roi(fmri_data, mask_data)
        self.assertEqual(output.shape[0], 8)

        fmri_data = np.random.rand(5, 13, 24)
        output = isc_fmri_roi(fmri_data, mask_data)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()
