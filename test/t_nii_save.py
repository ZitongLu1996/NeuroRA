# -*- coding: utf-8 -*-

' a module for testing neurora.nii_save module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.nii_save import corr_save_nii, stats_save_nii

class test_nii_save(unittest.TestCase):

    def test_corr_save_nii(self):

        corrs = np.random.rand(51, 61, 51, 2)
        affine = np.random.rand(3, 3)
        output = corr_save_nii(corrs, affine=affine, size=[53, 63, 53])
        self.assertEqual(output, "Invalid input!")

        corrs = np.random.rand(51, 61, 51)
        output = corr_save_nii(corrs, affine=affine, size=[53, 63, 53])
        self.assertEqual(output, "Invalid input!")

    def test_stats_save_nii(self):

        stats = np.random.rand(51, 61, 51, 2)
        affine = np.random.rand(3, 3)
        output = stats_save_nii(stats, affine=affine, size=[53, 63, 53])
        self.assertEqual(output, "Invalid input!")

        stats = np.random.rand(51, 61, 51)
        output = stats_save_nii(stats, affine=affine, size=[53, 63, 53])
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()