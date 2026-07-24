# -*- coding: utf-8 -*-

' a module for testing neurora.rsa_plot module '

__author__ = 'Zitong Lu'

import os
import numpy as np
import unittest
from neurora.rsa_plot import plot_rdm, plot_rdm_withvalue, plot_corrs_by_time, plot_tbytsim_withstats
from neurora.rsa_plot import plot_corrs_hotmap, plot_corrs_hotmap_stats, plot_nps_hotmap, plot_stats_hotmap
from neurora.rsa_plot import plot_brainrsa_regions, plot_brainrsa_montage, plot_brainrsa_glass, plot_brainrsa_surface, \
    plot_brainrsa_rlts

class test_rsa_plot(unittest.TestCase):

    def test_plot_rdm(self):

        rdm = np.random.rand(8, 8)
        output = plot_rdm(rdm)
        self.assertEqual(output, 0)

        rdm = np.random.rand(7, 8)
        output = plot_rdm(rdm)
        self.assertEqual(output, "Invalid input!")

    def test_plot_rdm_withvalue(self):

        rdm = np.random.rand(8, 8)
        output = plot_rdm(rdm)
        self.assertEqual(output, 0)

        rdm = np.random.rand(7, 8)
        output = plot_rdm_withvalue(rdm)
        self.assertEqual(output, "Invalid input!")

    def test_plot_corrs_by_time(self):

        corrs = np.random.rand(100, 5, 2)
        output = plot_corrs_by_time(corrs)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5)
        output = plot_corrs_by_time(corrs)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5, 2, 2)
        output = plot_corrs_by_time(corrs)
        self.assertEqual(output, "Invalid input!")

    def test_plot_tbytsim_withstats(self):

        Similarities = np.random.rand(20, 10, 2)
        output = plot_tbytsim_withstats(Similarities)
        self.assertEqual(output, 0)

        Similarities = np.random.rand(20, 10)
        output = plot_tbytsim_withstats(Similarities)
        self.assertEqual(output, 0)

        Similarities = np.random.rand(20, 10, 2, 2)
        output = plot_tbytsim_withstats(Similarities)
        self.assertEqual(output, "Invalid input!")

    def test_plot_corrs_hotmap(self):

        corrs = np.random.rand(100, 5, 2)
        output = plot_corrs_hotmap(corrs)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5)
        output = plot_corrs_hotmap(corrs)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5, 2, 2)
        output = plot_corrs_hotmap(corrs)
        self.assertEqual(output, "Invalid input!")

    def test_plot_corrs_hotmap_stats(self):

        stats = np.random.rand(100, 5, 2)
        corrs = np.random.rand(100, 5, 2)
        output = plot_corrs_hotmap_stats(corrs, stats)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5)
        output = plot_corrs_hotmap_stats(corrs, stats)
        self.assertEqual(output, 0)

        corrs = np.random.rand(100, 5, 2, 2)
        output = plot_corrs_hotmap_stats(corrs, stats)
        self.assertEqual(output, "Invalid input!")

    def test_plot_nps_hotmap(self):

        similarities = np.random.rand(10, 2)
        output = plot_nps_hotmap(similarities)
        self.assertEqual(output, 0)

        similarities = np.random.rand(10, 2, 2)
        output = plot_nps_hotmap(similarities)
        self.assertEqual(output, "Invalid input!")

    def test_plot_stats_hotmap(self):

        similarities = np.random.rand(5, 10, 2)
        output = plot_stats_hotmap(similarities)
        self.assertEqual(output, 0)

        similarities = np.random.rand(5, 10, 2, 2)
        output = plot_stats_hotmap(similarities)
        self.assertEqual(output, "Invalid input!")

    def test_plot_brainrsa_regions(self):

        img = '../neurora/template/ch2.nii.gz'
        output = plot_brainrsa_regions(img)
        self.assertEqual(output, 0)

    def test_plot_brainrsa_montage(self):

        img = '../neurora/template/ch2.nii.gz'
        output = plot_brainrsa_montage(img)
        self.assertEqual(output, 0)

    def test_plot_brainrsa_glass(self):

        img = '../neurora/template/ch2.nii.gz'
        output = plot_brainrsa_glass(img)
        self.assertEqual(output, 0)

    def test_plot_brainrsa_surface(self):

        img = '../neurora/template/ch2.nii.gz'
        output = plot_brainrsa_surface(img)
        self.assertEqual(output, 0)

    def test_plot_brainrsa_rlts(self):

        img = '../neurora/template/ch2.nii.gz'
        output = plot_brainrsa_rlts(img)
        self.assertEqual(output, 0)

if __name__ == '__main__':
    unittest.main()