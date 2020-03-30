# -*- coding: utf-8 -*-

' a module for saving the correlation coefficients in a .nii file '

__author__ = 'Zitong Lu'

import numpy as np
import nibabel as nib
import math

def corr_save_nii(corrs, filename, affine, size=[60, 60, 60], ksize=[3, 3, 3], strides=[1, 1, 1], p=1, r=0, similarity=0, distance=0):

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

    img = np.full([nx, ny, nz, 2, kx*ky*kz], np.nan)

    index = np.zeros([nx, ny, nz], dtype=np.int)

    img_nii = np.zeros([nx, ny, nz], dtype=np.float64)

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                x = i*sx
                y = j*sy
                z = k*sz

                if (math.isnan(corrs[i, j, k, 0]) == False) and (corrs[i, j, k, 1] < p) and (corrs[i, j, k, 0] > np.max(r, similarity, distance)):

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):

                                index[x+k1, y+k2, z+k3] = index[x+k1, y+k2, z+k3] + 1

    for i in range(n_x):

        for j in range(n_y):

            for k in range(n_z):

                x = i*sx
                y = i*sy
                z = i*sz

                if (math.isnan(corrs[i, j, k, 0]) == False) and (corrs[i, j, k, 1] < p) and (corrs[i, j, k, 0] > np.max(r, similarity, distance)):

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):

                                img_nii[x+k1, y+k2, z+k3] = img_nii[x+k1, y+k2, z+k3] + corrs[i, j, k]

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                img_nii[i, j, k] = img_nii[i, j, k]/index[i, j, k]

                """for k1 in range(kx):

                    for k2 in range(ky):

                        for k3 in range(kz):

                            index = 0

                            while math.isnan(img[x+k1, y+k2, z+k3, 0, index]) == False:

                                index = index + 1

                            img[x+k1, y+k2, z+k3, 0, index] = corrs[i, j, k, 0]
                            img[x+k1, y+k2, z+k3, 1, index] = corrs[i, j, k, 1]
                            #print(img[x+k1, y+k2, z+k3])

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                index = 0
                sum1 = 0
                sum2 = 0

                while img[i, j, k, 0, index] != 0:

                    sum1 = sum1 + img[i, j, k, 0, index]
                    sum2 = sum2 + img[i, j, k, 1, index]
                    if index == kx*ky*kz-1:
                        break
                    index = index + 1

                if index != 0:
                    rv = sum1 / index
                    pv = sum2 / index

                if index == 0:
                    rv = img[i, j, k, 0, 0]
                    pv = img[i, j, k, 1, 0]

                if p > 1:

                    return None

                elif p < 1:

                    if similarity > 0:

                        return None

                    elif distance > 0:

                        return None

                    elif r < 0:

                        return None

                    elif r >= 0:

                        if (pv < p) and (rv > r):

                            img_nii[i, j, k] = rv

                elif similarity > 0:

                    if distance > 0:

                        return None

                    if r > 0:

                        return None

                    elif r == 0:

                        if rv > similarity:

                            img_nii[i, j, k] = rv

                elif distance > 0:

                    if r > 0:

                        return None

                    elif r == 0:

                        if rv > distance:

                            img_nii[i, j, k] = rv"""

    filename = filename+".nii"

    print(filename)

    file = nib.Nifti1Image(img_nii, affine)

    nib.save(file, filename)

    print("File("+filename+") saves successfully!")

    return img_nii