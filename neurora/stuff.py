# -*- coding: utf-8 -*-

' a module for some simple but important processes '

__author__ = 'Zitong Lu'

import nibabel as nib
import numpy as np

def limtozero(x):
    if x < 1e-15:
        x = 0
    return x

def get_affine(file_name):

    img = nib.load(file_name)

    return img.affine

def fwe_correct(p, size=[60, 60, 60], n=64):

    x = size[0]
    y = size[1]
    z = size[2]

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    n = float(n*px*py*pz/(x*y*z))

    nq = 1
    ni = 1
    while nq < n:
        ni = ni + 1
        nq = ni*ni*ni

    n = nq

    print(n, ni)

    fwep = p*n

    print("finished FWE correct")

    return fwep

def fdr_correct(p, size=[60, 60, 60], n=64):

    x = size[0]
    y = size[1]
    z = size[2]

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    n = float(n*px*py*pz/(x*y*z))

    nq = 1
    ni = 1
    while nq < n:
        ni = ni + 1
        nq = ni*ni*ni

    n = nq

    print(n, ni)

    fdrp = np.full([px, py, pz], np.nan)

    for i in range(px-ni+1):
        for j in range(py-ni+1):
            for k in range(pz-ni+1):

                pcluster = p[i:i+ni, j:j+ni, k:k+ni]
                pcluster = np.reshape(pcluster, [n])

                index = np.argsort(pcluster)
                for l in range(n):
                    pcluster[l] = float(pcluster[l]*n/(l+1))

                for l in range(n-1):
                    if pcluster[-l-1] < pcluster[-l-2]:
                        pcluster[-l-2] = pcluster[-l-1]

                newpcluster = np.full([n], np.nan)
                for l in range(n):
                    newpcluster[l] = pcluster[index[l]]

                fdrp[i:i+ni, j:j+ni, k:k+ni] = np.reshape(newpcluster, [ni, ni, ni])

    print("finished FDR correct")

    return fdrp