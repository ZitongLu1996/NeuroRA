# -*- coding: utf-8 -*-

' a module for testing neurora.rdm_corr module '

__author__ = 'Zitong Lu'

import numpy as np
import unittest
from neurora.rdm_corr import rdm_correlation_pearson, rdm_correlation_spearman, rdm_correlation_kendall, rdm_distance,\
    rdm_similarity

class test_rdm_corr(unittest.TestCase):

    def test_rdm_correlation_pearson(self):

        rdm1 = np.random.rand(8, 8)
        rdm2 = np.random.rand(8, 8)
        rp = rdm_correlation_pearson(rdm1, rdm2, permutation=True)
        self.assertEqual(len(rp), 2)

        rp = rdm_correlation_pearson(rdm1, rdm2, permutation=False)
        self.assertEqual(len(rp), 2)

        rdm1 = np.random.rand(8, 7)
        rp = rdm_correlation_pearson(rdm1, rdm2)
        self.assertEqual(rp, "Invalid input!")

    def test_rdm_correlation_spearman(self):

        rdm1 = np.random.rand(8, 8)
        rdm2 = np.random.rand(8, 8)
        rp = rdm_correlation_spearman(rdm1, rdm2, permutation=True)
        self.assertEqual(len(rp), 2)

        rp = rdm_correlation_spearman(rdm1, rdm2, permutation=False)
        self.assertEqual(len(rp), 2)

        rdm1 = np.random.rand(8, 7)
        rp = rdm_correlation_spearman(rdm1, rdm2)
        self.assertEqual(rp, "Invalid input!")

    def test_rdm_correlation_kendall(self):

        rdm1 = np.random.rand(8, 8)
        rdm2 = np.random.rand(8, 8)
        rp = rdm_correlation_kendall(rdm1, rdm2, permutation=True)
        self.assertEqual(len(rp), 2)

        rp = rdm_correlation_kendall(rdm1, rdm2, permutation=False)
        self.assertEqual(len(rp), 2)

        rdm1 = np.random.rand(8, 7)
        rp = rdm_correlation_kendall(rdm1, rdm2)
        self.assertEqual(rp, "Invalid input!")

    def test_rdm_rdm_distance(self):

        rdm1 = np.random.rand(8, 8)
        rdm2 = np.random.rand(8, 8)
        rp = rdm_distance(rdm1, rdm2)
        self.assertIsNotNone(rp)

        rdm1 = np.random.rand(8, 7)
        rp = rdm_distance(rdm1, rdm2)
        self.assertEqual(rp, "Invalid input!")

    def test_rdm_similarity(self):

        rdm1 = np.random.rand(8, 8)
        rdm2 = np.random.rand(8, 8)
        rp = rdm_similarity(rdm1, rdm2)
        self.assertIsNotNone(rp)

        rdm1 = np.random.rand(8, 7)
        rp = rdm_similarity(rdm1, rdm2)
        self.assertEqual(rp, "Invalid input!")

if __name__ == '__main__':
    unittest.main()