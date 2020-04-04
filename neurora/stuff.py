# -*- coding: utf-8 -*-

' a module for some simple but important processes '

__author__ = 'Zitong Lu'

import nibabel as nib
import numpy as np
import os

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

def correct_by_threshold(img, threshold):

    sx = np.shape(img)[0]
    sy = np.shape(img)[1]
    sz = np.shape(img)[2]

    nsmall = 1
    while nsmall*nsmall*nsmall < threshold:
        nsmall = nsmall + 1

    nlarge = nsmall + 2

    for i in range(sx-nlarge+1):
        for j in range(sy-nlarge+1):
            for k in range(sz-nlarge+1):

                listlarge = list(np.reshape(img[i:i+nlarge, j:j+nlarge, k:k+nlarge], [nlarge*nlarge*nlarge]))
                print(listlarge.count(0))
                if listlarge.count(0) < nlarge*nlarge*nlarge:
                    index1 = 0
                    for l in range(nlarge):
                        for m in range(nlarge):
                            if img[i + l, j + m, k] == 0:
                                index1 = index1 + 1
                            if img[i + l, j + m, k + nlarge - 1] == 0:
                                index1 = index1 + 1
                    for l in range(nlarge-1):
                        for m in range(nlarge-2):
                            if img[i + l, j, k + m] == 0:
                                index1 = index1 + 1
                            if img[i, j + l + 1, k + m] == 0:
                                index1 = index1 + 1
                            if img[i + nlarge - 1, j + l, k + m] == 0:
                                index1 = index1 + 1
                            if img[i + l + 1, j + nlarge - 1, k + m] == 0:
                                index1 = index1 + 1
                    nex = nlarge * nlarge * nlarge - nsmall * nsmall * nsmall
                    print("index1:"+str(index1))
                    if index1 == nex:
                        print("**************************")
                        unit = img[i+1:i+1+nsmall, j+1:j+1+nsmall, k+1:k+1+nsmall]
                        unit = np.reshape(unit, [nsmall*nsmall*nsmall])
                        list_internal = list(unit)
                        index2 = nsmall*nsmall*nsmall-list_internal.count(0)
                        print(index1, index2)
                        if index2 < threshold:
                            img[i+1:i+1+nsmall, j]
                            for l in range(nsmall):
                                for m in range(nsmall):
                                    for p in range(nsmall):
                                        img[i+1:i+1+nsmall, j+1:j+1+nsmall, k+1:k+1+nsmall] = np.zeros([nsmall, nsmall, nsmall])

    print("finished correction")

    return img

def get_bg_ch2():

    return os.path.abspath('./template/ch2.nii.gz')

def get_bg_ch2bet():

    return os.path.abspath('./template/ch2bet.nii.gz')