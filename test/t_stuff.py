# -*- coding: utf-8 -*-

' a module for testing neurora.stuff module '

__author__ = 'Zitong Lu'

import os
import numpy as np
import unittest
from neurora.stuff import limtozero, get_affine, fisherz_rdm, fwe_correct, fdr_correct, correct_by_threshold, \
    get_bg_ch2, get_bg_ch2bet, get_HOcort, datamask, position_to_mni, mask_to, permutation_test, permutation_corr

class test_stuff(unittest.TestCase):

    def test_limtozero(self):

        x = 0.1*1e-15
        output = limtozero(x)
        self.assertEqual(output, 0)

        x = 2
        output = limtozero(x)
        self.assertEqual(output, 2)

    def test_get_affine(self):

        img = '../neurora/template/ch2.nii.gz'
        output = get_affine(img)
        self.assertEqual(output.shape[0], 4)

    def test_fisherz_rdm(self):

        rdm = np.random.rand(8, 8)
        output = fisherz_rdm(rdm)
        self.assertEqual(output.shape[0], 8)

        rdm = np.random.rand(7, 8)
        output = fisherz_rdm(rdm)
        self.assertEqual(output, "Invalid input!")

    def test_fwe_correct(self):

        p = np.random.rand(20, 22, 21)
        p_threshold = 0.05
        output = fwe_correct(p, p_threshold)
        self.assertEqual(output.shape[0], 20)

    def test_fdr_correct(self):

        p = np.random.rand(20, 22, 21)
        p_threshold = 0.05
        output = fdr_correct(p, p_threshold)
        self.assertEqual(output.shape[0], 20)

    def test_correct_by_threshold(self):

        img = np.random.rand(20, 22, 21)
        threshold = 25
        output = correct_by_threshold(img, threshold)
        self.assertEqual(output.shape[0], 20)

    def test_get_bg_ch2(self):

        output = get_bg_ch2()
        package_root = os.path.dirname(os.path.abspath(__file__))
        self.assertEqual(output, os.path.join(package_root, 'template/ch2.nii.gz'))

    def test_get_bg_ch2bet(self):

        output = get_bg_ch2bet()
        package_root = os.path.dirname(os.path.abspath(__file__))
        self.assertEqual(output, os.path.join(package_root, 'template/ch2bet.nii.gz'))

    def test_get_HOcort(self):

        output = get_HOcort()
        package_root = os.path.dirname(os.path.abspath(__file__))
        self.assertEqual(output, os.path.join(package_root, 'template/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz'))

    def test_datamask(self):

        fmri_data = np.random.rand(13, 14, 12)
        mask_data = np.random.rand(13, 14, 12)
        output = datamask(fmri_data, mask_data)
        self.assertEqual(output.shape[0], 12)

    def test_position_to_mni(self):

        point = [2, 5, 6]
        affine = np.random.rand(4, 4)
        output = position_to_mni(point, affine)
        self.assertEqual(len(output), 3)

    def test_mask_to(self):

        mask = '../neurora/template/ch2.nii.gz'
        size = [57, 67, 56]
        affine = np.array([[3, 0, 0, -78],
                           [0, 2.866, -0.887, -76],
                           [0, 0.887, 2.866, -64],
                           [0, 0, 0, 1]])
        output = mask_to(mask, size, affine)
        self.assertEqual(output, 0)

    def test_permutation_test(self):

        v1 = np.random.rand(20)
        v2 = np.random.rand(20)
        output = permutation_test(v1, v2)
        self.assertIsNotNone(output)

    def test_permutation_corr(self):

        v1 = np.random.rand(20)
        v2 = np.random.rand(20)
        output = permutation_corr(v1, v2)
        self.assertIsNotNone(output)

if __name__ == '__main__':
    unittest.main()