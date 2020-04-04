# -*- coding: utf-8 -*-

' a module for saving the correlation coefficients in a .nii file '

__author__ = 'Zitong Lu'

import numpy as np
import nibabel as nib
import math
from neurora.stuff import fwe_correct, fdr_correct
from neurora.rsa_plot import plot_brainrsa_rlts

def corr_save_nii(corrs, filename, affine, size=[60, 60, 60], ksize=[3, 3, 3], strides=[1, 1, 1], p=1, r=0, similarity=0, distance=0, correct_method=None, correct_n=27, plotrlt=True, img_background=None):

    nx = size[0]
    ny = size[1]
    nz = size[2]

    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    n_x = np.shape(corrs)[0]
    n_y = np.shape(corrs)[1]
    n_z = np.shape(corrs)[2]

    index = np.zeros([nx, ny, nz], dtype=np.int)

    img_nii = np.zeros([nx, ny, nz], dtype=np.float64)

    nfdr = 0

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                x = i*sx
                y = j*sy
                z = k*sz

                if (math.isnan(corrs[i, j, k, 0]) is False):

                    nfdr = nfdr + 1

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):

                                index[x+k1, y+k2, z+k3] = index[x+k1, y+k2, z+k3] + 1

    mask = np.zeros([nx, ny, nz], dtype=np.int)

    corrsp = corrs[:, :, :, 1]

    if p < 1:

        if correct_method == "FDR":

            corrsp = fdr_correct(corrsp, size=size, n=correct_n)

        if correct_method == "FWE":

            corrsp = fwe_correct(corrsp, size=size, n=correct_n)

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                x = i * sx
                y = j * sy
                z = k * sz

                if (corrsp[i, j, k] < p) and (corrs[i, j, k, 0] > np.max([r, similarity, distance])):

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):
                                mask[x + k1, y + k2, z + k3] = 1

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                x = i*sx
                y = j*sy
                z = k*sz

                if (math.isnan(corrs[i, j, k, 0]) is False):

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):

                                img_nii[x+k1, y+k2, z+k3] = img_nii[x+k1, y+k2, z+k3] + corrs[i, j, k, 0]

    newimg_nii = np.full([nx, ny, nz], np.nan)
    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                if mask[i, j, k] == 1:
                    newimg_nii[i, j, k] = float(img_nii[i, j, k]/index[i, j, k])

    filename = filename+".nii"

    print(filename)

    file = nib.Nifti1Image(newimg_nii, affine)

    nib.save(file, filename)

    if plotrlt == True:

        plot_brainrsa_rlts(file, background=img_background)

    print("File("+filename+") saves successfully!")

    return newimg_nii