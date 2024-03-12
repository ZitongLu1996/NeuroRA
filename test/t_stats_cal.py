# -*- coding: utf-8 -*-

' a module for testing neurora.stats_cal module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.stats_cal import stats, stats_fmri, stats_iscfmri, stats_stps, stats_stpsfmri

class test_stats_cal(unittest.TestCase):

    def test_stats(self):

        corrs = np.random.rand(10, 32, 50, 2)
        output = stats(corrs, permutation=False)
        self.assertEqual(output.shape[0], 32)

        corrs = np.random.rand(10, 32, 50)
        output = stats(corrs, permutation=False)
        self.assertEqual(output, "Invalid input!")

    def test_stats_fmri(self):

        corrs = np.random.rand(8, 13, 14, 12, 2)
        output = stats_fmri(corrs, permutation=False)
        self.assertEqual(output.shape[0], 13)

        corrs = np.random.rand(8, 13, 14, 12)
        output = stats_fmri(corrs, permutation=False)
        self.assertEqual(output, "Invalid input!")

    def test_stats_iscfmri(self):

        corrs = np.random.rand(20, 10, 13, 14, 12, 2)
        output = stats_iscfmri(corrs, permutation=False)
        self.assertIsNotNone(output.shape[0], 20)

        corrs = np.random.rand(13, 14, 12)
        output = stats_iscfmri(corrs, permutation=False)
        self.assertEqual(output, "Invalid input!")

    def test_stats_stps(self):

        corrs1 = np.random.rand(10, 5, 20)
        corrs2 = np.random.rand(10, 5, 20)
        output = stats_stps(corrs1, corrs2, permutation=False)
        self.assertEqual(output.shape[0], 5)
        
        corrs1 = np.random.rand(10, 4, 20)
        output = stats_stps(corrs1, corrs2, permutation=False)
        self.assertEqual(output, "Invalid input!")

    def test_stats_stpsfmri(self):
        
        corrs1 = np.random.rand(8, 13, 14, 12)
        corrs2 = np.random.rand(8, 13, 14, 12)
        output = stats_stpsfmri(corrs1, corrs2, permutation=False)
        self.assertEqual(output.shape[0], 13)
        
        corrs1 = np.random.rand(8, 13, 13, 12)
        output = stats_stpsfmri(corrs1, corrs2, permutation=False)
        self.assertEqual(output, "Invalid input!")

if __name__ == '__main__':
    unittest.main()