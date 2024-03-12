# -*- coding: utf-8 -*-

' a module for some simple but important processes '

__author__ = 'Zitong Lu'

import nibabel as nib
import numpy as np
import os
import math
from scipy.stats import spearmanr, pearsonr, kendalltau, ttest_1samp, ttest_rel
from skimage.measure import label
import sys

# get package abspath
package_root = os.path.dirname(os.path.abspath(__file__))


' a function for zeroing the value close to zero '

def limtozero(x):

    """
    zero the value close to zero

    Parameters
    ----------
    x : float

    Returns
    -------
    0
    """

    if x < 1e-15:
        x = 0

    return x


' a function for getting the affine of the fMRI-img '

def get_affine(file_name):

    """
    get the affine of the fMRI-img

    Parameters
    ----------
    file_name : string
        The filename of a sample fMRI-img in your experiment

    Returns
    -------
    affine : array
        The position information of the fMRI-image array data in a reference space.
    """

    if file_name == "":

        return "Invalid input"

    img = nib.load(file_name)

    return img.affine


' a function for fMRI RSA results correction by threshold '

def correct_by_threshold(img, threshold):

    """
    correct the fMRI RSA results by threshold

    Parameters
    ----------
    img : array
        A 3-D array of the fMRI RSA results.
        The shape of img should be [nx, ny, nz]. nx, ny, nz represent the shape of the fMRI-img.
    threshold : int
        The number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than n voxels will be visualized.

    Returns
    -------
    img : array
        A 3-D array of the fMRI RSA results after correction.
        The shape of img should be [nx, ny, nz]. nx, ny, nz represent the shape of the fMRI-img.
    """

    if len(np.shape(img)) != 3:

        return "Invalid input"

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

                    if index1 == nex:
                        unit = img[i+1:i+1+nsmall, j+1:j+1+nsmall, k+1:k+1+nsmall]
                        unit = np.reshape(unit, [nsmall*nsmall*nsmall])
                        list_internal = list(unit)
                        index2 = nsmall*nsmall*nsmall-list_internal.count(0)

                        if index2 < threshold:
                            img[i+1:i+1+nsmall, j]

                            for l in range(nsmall):
                                for m in range(nsmall):
                                    for p in range(nsmall):
                                        img[i+1:i+1+nsmall, j+1:j+1+nsmall, k+1:k+1+nsmall] = np.zeros([nsmall, nsmall, nsmall])

    print("finished correction")

    return img


' a function for FWE-correction for fMRI RSA results '

def fwe_correct(p, p_threshold):

    """
    FWE correction for fMRI RSA results

    Parameters
    ----------
    p : array
        The p-value map (3-D).
    p_threshold: string
        The p threshold.

    Returns
    -------
    fwep : array.
        The FWE corrected p-value map.
    """

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    n = 0

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if (math.isnan(p[i, j, k]) == False):
                    n = n + 1

    p = p*n

    fwep = np.full([px, py, pz], np.nan)

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if p[i, j, k] < p_threshold:
                        fwep[i, j, k] = p[i, j, k]

    print("finished FWE correct")

    return fwep


' a function for FDR-correction for fMRI RSA results '

def fdr_correct(p, p_threshold):

    """
    FDR correction for fMRI RSA results

    Parameters
    ----------
    p : array
        The p-value map (3-D).
    p_threshold: string
        The p threshold.

    Returns
    -------
    fdrp : array.
        The FDR corrected p-value map.
    """

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    n = 0

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if (math.isnan(p[i, j, k]) == False):

                    n = n + 1

    fdrp = np.full([px, py, pz], np.nan)

    pcluster = np.zeros([n])

    m = 0

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if (math.isnan(p[i, j, k]) == False):
                    pcluster[m] = p[i, j, k]
                    m = m + 1

    index = np.argsort(pcluster)

    for l in range(n):
        pcluster[index[l]] = float(pcluster[index[l]]*n/(l+1))

    """for l in range(n - 1):
        if pcluster[index[-l - 1]] < pcluster[index[-l - 2]]:
            pcluster[index[-l - 2]] = pcluster[index[-l - 1]]"""

    m = 0

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if math.isnan(p[i, j, k]) == False:
                    if p[i, j, k] < p_threshold:
                        fdrp[i, j, k] = pcluster[m]
                    m = m + 1

    print("finished FDR correct")

    return fdrp


' a function for Cluster-wise FWE-correction for fMRI RSA results '

def cluster_fwe_correct(p, p_threshold1, p_threshold2):

    """
    Cluster-wise FWE correction for fMRI RSA results

    Parameters
    ----------
    p : array
        The p-value map (3-D).
    p_threshold1: string
        The voxel-wise p threshold.
    p_threshold2: string
        The cluster-wise p threshold

    Returns
    -------
    clusterfwep : array.
        The Cluster-wise FWE corrected p-value map.
    """

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    p01 = np.zeros([px, py, pz])

    for i in range(px):
        for j in range(py):
            for k in range(pz):
                if p[i, j, k] < p_threshold1:
                    p01[i, j, k] = 1

    print("Cluster-wise FWE correction")

    permutation_voxels = np.zeros([1000])
    for k in range(1000):

        # show the progressbar
        percent = (k+1) / 1000 * 100
        show_progressbar("Correcting", percent)

        pi = np.copy(p01)
        pi = np.reshape(pi, [px * py * pz])
        indexk = np.arange(0, px * py * pz)
        np.random.shuffle(indexk)
        pi = pi[indexk]
        pi = np.reshape(pi, [px, py, pz])
        labels = label(pi, connectivity=1)
        nclusters = int(np.max(labels))
        voxelsinluster = np.zeros([nclusters + 1], dtype=int)
        labels = np.reshape(labels, [px * py * pz])
        for i in range(px * py * pz):
            voxelsinluster[labels[i]] = voxelsinluster[labels[i]] + 1
        permutation_voxels[k] = max(voxelsinluster[1:])

    print("\n")

    permutation_voxels = np.sort(permutation_voxels)
    voxels_threshold = permutation_voxels[int(1000*(1-p_threshold2))]

    labels = label(p01, connectivity=1)
    nclusters = int(np.max(labels))
    voxelsinluster = np.zeros([nclusters + 1], dtype=int)
    labels = np.reshape(labels, [px * py * pz])
    for i in range(px * py * pz):
        voxelsinluster[labels[i]] = voxelsinluster[labels[i]] + 1
    voxelsinluster = voxelsinluster[1:]
    labels = np.reshape(labels, [px, py, pz])

    clusterp = np.zeros([nclusters])
    for i in range(nclusters):
        clusterp[i] = (1000 - np.max(np.array(np.where(np.sort(np.append(permutation_voxels, voxelsinluster[i])) == voxelsinluster[i])))) / 1000

    clusterp = clusterp * nclusters

    clusterfwep = np.full([px, py, pz], np.nan)

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if (math.isnan(p[i, j, k]) == False) and labels[i, j, k] != 0\
                        and clusterp[labels[i, j, k]-1] < p_threshold1 and voxelsinluster[labels[i, j, k]-1] >= voxels_threshold:
                    clusterfwep[i, j, k] = clusterp[labels[i, j, k]-1]

    print("finished Cluster-wise FWE correction")

    return clusterfwep


' a function for Cluster-wise FDR-correction for fMRI RSA results '

def cluster_fdr_correct(p, p_threshold1, p_threshold2):

    """
    Cluster-wise FDR correction for fMRI RSA results

    Parameters
    ----------
    p : array
        The p-value map (3-D).
    p_threshold1: string
        The voxel-wise p threshold.
    p_threshold2: string
        The cluster-wise p threshold

    Returns
    -------
    clusterfdrp : array.
        The Cluster-wise FDR corrected p-value map.
    """

    px = np.shape(p)[0]
    py = np.shape(p)[1]
    pz = np.shape(p)[2]

    p01 = np.zeros([px, py, pz])

    for i in range(px):
        for j in range(py):
            for k in range(pz):
                if p[i, j, k] < p_threshold1:
                    p01[i, j, k] = 1

    print("Cluster-wise FDR correction")

    permutation_voxels = np.zeros([1000])
    for k in range(1000):

        # show the progressbar
        percent = (k+1) / 1000 * 100
        show_progressbar("Correcting", percent)

        pi = np.copy(p01)
        pi = np.reshape(pi, [px * py * pz])
        indexk = np.arange(0, px * py * pz)
        np.random.shuffle(indexk)
        pi = pi[indexk]
        pi = np.reshape(pi, [px, py, pz])
        labels = label(pi, connectivity=1)
        nclusters = int(np.max(labels))
        voxelsinluster = np.zeros([nclusters+1], dtype=int)
        labels = np.reshape(labels, [px*py*pz])
        for i in range(px*py*pz):
            voxelsinluster[labels[i]] = voxelsinluster[labels[i]] + 1
        permutation_voxels[k] = max(voxelsinluster[1:])

    print("\n")

    permutation_voxels = np.sort(permutation_voxels)
    voxels_threshold = permutation_voxels[int(1000*(1-p_threshold2))]

    labels = label(p01, connectivity=1)
    nclusters = int(np.max(labels))
    voxelsinluster = np.zeros([nclusters+1], dtype=int)
    labels = np.reshape(labels, [px*py*pz])
    for i in range(px*py*pz):
        voxelsinluster[labels[i]] = voxelsinluster[labels[i]] + 1
    voxelsinluster = voxelsinluster[1:]
    labels = np.reshape(labels, [px, py, pz])

    clusterp = np.zeros([nclusters])
    for i in range(nclusters):
        clusterp[i] = (1000 - np.max(np.array(np.where(np.sort(np.append(permutation_voxels, voxelsinluster[i])) ==
                                                       voxelsinluster[i])))) / 1000

    index = np.argsort(clusterp)

    for i in range(nclusters):
        clusterp[index[i]] = float(clusterp[index[i]] * nclusters / (i+1))

    clusterfdrp = np.full([px, py, pz], np.nan)

    for i in range(px):
        for j in range(py):
            for k in range(pz):

                if (math.isnan(p[i, j, k]) == False) and labels[i, j, k] != 0\
                        and clusterp[labels[i, j, k]-1] < p_threshold1 and voxelsinluster[labels[i, j, k]-1] >= \
                        voxels_threshold:
                    clusterfdrp[i, j, k] = clusterp[labels[i, j, k]-1]

    print("finished Cluster-wise FDR correction")

    return clusterfdrp


' a function for getting ch2.nii.gz '

def get_bg_ch2():

    """
    get ch2.nii.gz

    Returns
    -------
    path : string
        The absolute file path of 'ch2.nii.gz'
    """

    return os.path.join(package_root, 'template/ch2.nii.gz')


' a function for getting ch2bet.nii.gz '

def get_bg_ch2bet():

    """
    get ch2bet.nii.gz

    Returns
    -------
    path : string
        The absolute file path of 'ch2bet.nii.gz'
    """

    return os.path.join(package_root, 'template/ch2bet.nii.gz')


' a function for getting HarvardOxford-cort-maxprob-thr0-1mm.nii.gz '

def get_HOcort():
    """
    get HarvardOxford-cort-maxprob-thr0-1mm.nii.gz

    Returns
    -------
    path : string
        The absolute file path of 'HarvardOxford-cort-maxprob-thr0-1mm.nii.gz'
    """

    return os.path.join(package_root, 'template/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz')


' a function for filtering the data by a ROI mask '

def datamask(fmri_data, mask_data):

    """
    filter the data by a ROI mask

    Parameters
    ----------
    fmri_data : array
        The fMRI data.
        The shape of fmri_data is [nx, ny, nz]. nx, ny, nz represent the size of the fMRI data.
    mask_data : array
        The mask data.
        The shape of mask_data is [nx, ny, nz]. nx, ny, nz represent the size of the fMRI data.

    Returns
    -------
    newfmri_data : array
        The new fMRI data.
        The shape of newfmri_data is [nx, ny, nz]. nx, ny, nz represent the size of the fMRI data.
    """

    if len(np.shape(fmri_data)) != 3 or len(np.shape(mask_data)) != 3:

        return "Invalid input"

    nx, ny, nz = fmri_data.shape

    newfmri_data = np.full([nx, ny, nz], np.nan)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) is False):
                    newfmri_data[i, j, k] = fmri_data[i, j, k]

    return newfmri_data


' a function for projecting the position of a point in matrix coordinate system to the position in MNI coordinate system '

def position_to_mni(point, affine):

    """
    project the position in matrix coordinate system to the position in MNI coordinate system

    Parameters
    ----------
    point : list or array
        The position in matrix coordinate system.
    affine : array or list
        The position information of the fMRI-image array data in a reference space.

    Returns
    -------
    newpoint : array
        The position in MNI coordinate system.
    """

    i = point[0]
    j = point[1]
    k = point[2]

    x = -affine[0, 3] + i * affine[0, 0] - affine[0, 0]
    y = affine[1, 3] + j * affine[1, 1] - affine[1, 1]
    z = affine[2, 3] + k * affine[2, 2] - affine[2, 2]

    newpoint = np.array([x, y, z])

    return newpoint


' a function for project the position in MNI coordinate system to the position of a point in matrix coordinate system '

def mniposition_to(mnipoint, affine):

    """
    project the position in MNI coordinate system to the position of a point in matrix coordinate system

    Parameters
    ----------
    point : list or array
        The position in MNI coordinate system.
    affine : array or list
        The position information of the fMRI-image array data in a reference space.

    Returns
    -------
    newpoint : array
        The position in matrix coordinate system.
    """

    mx = int(float((mnipoint[0] - affine[0, 3])/affine[0, 0]))
    my = int(float((mnipoint[1] - affine[1, 3])/affine[1, 1]))
    mz = int(float((mnipoint[2] - affine[2, 3])/affine[2, 2]))

    return mx, my, mz


' a function for converting data of the mask template to your data template '

def mask_to(mask, size, affine, filename=None):

    """
    convert mask data of certain template to your data template

    Parameters
    ----------
    mask : string
        The file path+filename for the mask of certain template.
    size : array or list [nx, ny, nz]
        The size of the fMRI-img in your experiments.
    affine : array or list
        The position information of the fMRI-image array data in a reference space.
    filename : string. Default is None - 'newmask.nii'.
        The file path+filename for the mask for your data template .nii file.

    Notes
    -----
    A result .nii file of new mask will be generated at the corresponding address of filename.
    """

    data = nib.load(mask).get_fdata()

    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]

    maskaffine = nib.load(mask).affine

    newdata = np.zeros(size)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if data[i, j, k] != 0 or math.isnan(data[i, j, k]) is False:
                    mx = maskaffine[0, 3]+(i-1)*maskaffine[0, 0]
                    my = maskaffine[1, 3]+(j-1)*maskaffine[1, 1]
                    mz = maskaffine[2, 3]+(k-1)*maskaffine[2, 2]
                    x = int(float((mx-affine[0, 3])/affine[0, 0]))
                    y = int(float((my-affine[1, 3])/affine[1, 1]))
                    z = int(float((mz-affine[2, 3])/affine[2, 2]))
                    if x < size[0] and y < size[1] and z < size[2]:
                        newdata[x, y, z] = data[i, j, k]

    file = nib.Nifti1Image(newdata, affine)

    if filename == None:
        filename = "newmask.nii"
    else:
        q = ".nii" in filename

        if q == True:
            filename = filename
        else:
            filename = filename+".nii"

    nib.save(file, filename)

    return 0


' a function for permutation test '

def permutation_test(v1, v2, iter=1000):

    """
    Conduct Permutation test

    Parameters
    ----------
    v1 : array
        Vector 1.
    v2 : array
        Vector 2.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    if len(v1) != len(v2):

        return "Invalid input"

    # permutation test

    diff = abs(np.average(v1) - np.average(v2))
    v = np.hstack((v1, v2))
    nv = v.shape[0]
    ni = 0

    for i in range(iter):
        vshuffle = np.random.permutation(v)
        vshuffle1 = vshuffle[:int(nv/2)]
        vshuffle2 = vshuffle[int(nv/2):]
        diff_i = np.average(vshuffle1) - np.average(vshuffle2)

        if diff_i >= diff:
            ni = ni + 1

    # permunitation test p-value
    p = np.float64(ni/iter)

    return p


' a function for permutation test for correlation coefficients '

def permutation_corr(v1, v2, method="spearman", iter=1000):

    """
    Conduct Permutation test for correlation coefficients

    Parameters
    ----------
    v1 : array
        Vector 1.
    v2 : array
        Vector 2.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    if len(v1) != len(v2):

        return "Invalid input"

    # permutation test

    if method == "spearman":

        rtest = spearmanr(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = spearmanr(v1shuffle, v2shuffle)[0]

            if rperm > rtest:
                ni = ni + 1

    if method == "pearson":
        print(iter)
        rtest = pearsonr(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = pearsonr(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    if method == "kendalltau":

        rtest = kendalltau(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = kendalltau(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    p = np.float64((ni+1)/(iter+1))

    return p


' a function for getting the 1-D & 1-sided cluster-index information '

def get_cluster_index_1d_1sided(m):

    """
    Get 1-D & 1-sided cluster-index information from a vector

    Parameters
    ----------
    m : array
        A significant vector.
        The values in m should be 0 or 1, which represent not significant point or significant point, respectively.

    Returns
    -------
    index_v : array
        The cluster-index vector.
    index_n : int
        The number of clusters.
    """

    x = np.shape(m)[0]
    b = np.zeros([x+2])
    b[1:x+1] = m

    index_v = np.zeros([x])

    index_n = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] == 0 and b[i+2] == 1:
            index_n = index_n + 1
        if b[i+1] == 1:
            if b[i] != 0 or b[i+2] != 0:
                index_v[i] = index_n

    return index_v, index_n


' a function for getting the 1-D & 2-sided cluster-index information '

def get_cluster_index_1d_2sided(m):

    """
    Get 1-D & 2-sided cluster-index information from a vector

    Parameters
    ----------
    m : array
        A significant vector.
        The values in m should be 0 or 1 or -1, which represent not significant point or significantly higher point or
        significantly less point, respectively.

    Returns
    -------
    index_v1 : array
        The "greater" cluster-index vector.
    index_n1 : int
        The number of "greater" clusters.
    index_v2 : array
        The "less" cluster-index vector.
    index_n2 : int
        The number of "less" clusters.
    """

    x = np.shape(m)[0]
    b = np.zeros([x+2])
    b[1:x+1] = m

    index_v1 = np.zeros([x])

    index_n1 = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] != 1 and b[i+2] == 1:
            index_n1 = index_n1 + 1
        if b[i+1] == 1:
            if b[i] == 1 or b[i+2] == 1:
                index_v1[i] = index_n1

    index_v2 = np.zeros([x])

    index_n2 = 0
    for i in range(x):
        if b[i + 1] == -1 and b[i] != -1 and b[i + 2] == -1:
            index_n2 = index_n2 + 1
        if b[i + 1] == -1:
            if b[i] == -1 or b[i + 2] == -1:
                index_v2[i] = index_n2

    return index_v1, index_n1, index_v2, index_n2


' a function for getting the 2-D & 1-sided cluster-index information '

def get_cluster_index_2d_1sided(m):

    """
    Get 2-D & 1-sided cluster-index information from a matrix

    Parameters
    ----------
    m : array
        A significant matrix.
        The values in m should be 0 or 1, which represent not significant point or significant point, respectively.

    Returns
    -------
    index_m : array
        The cluster-index matrix.
    index_n : int
        The number of clusters.
    """

    x, y = np.shape(m)
    b = np.zeros([x+2, y+2])
    b[1:x+1, 1:y+1] = m

    index_m = np.zeros([x, y])

    index_n = 0
    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            if b[ii, jj] == 1 and (b[ii-1, jj]+b[ii+1, jj]+b[ii, jj-1]+b[ii, jj+1]) != 0:
                min_index = index_n + 1
                if b[ii - 1, jj] == 1:
                    min_index = np.min([min_index, index_m[i - 1, j]])
                if b[ii, jj - 1] == 1:
                    min_index = np.min([min_index, index_m[i, j - 1]])
                k1 = 0
                while b[ii, jj - k1] == 1:
                    index_m[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b[ii - k2, jj] == 1:
                    index_m[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b[ii, jj + k] == 1:
                    index_m[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b[ii + k, jj] == 1:
                    index_m[i + k, j] = min_index
                    k = k + 1
                if b[ii, jj - 1] != 1:
                    index_n = index_n + 1
                    k = 0
                    m = 0
                    while b[ii, jj + k] == 1:
                        if b[ii - 1, jj + k] == 1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n = index_n - 1

    return index_m, index_n


' a function for getting the 2-D & 2-sided cluster-index information '

def get_cluster_index_2d_2sided(m):

    """
    Get 2-D & 2-sided cluster-index information from a matrix

    Parameters
    ----------
    m : array
        A significant matrix.
        The values in m should be 0 or 1 or -1, which represent not significant point or significantly higher point or
        significantly less point, respectively.

    Returns
    -------
    index_m1 : array
        The "greater" cluster-index matrix.
    index_n1 : int
        The "greater" number of clusters.
    index_m2 : array
        The "less" cluster-index matrix.
    index_n2 : int
        The "less" number of clusters.
    """

    x, y = np.shape(m)
    b1 = np.zeros([x+2, y+2])
    b1[1:x+1, 1:y+1] = m
    b2 = np.zeros([x+2, y+2])
    b2[1:x+1, 1:y+1] = m
    index_m1 = np.zeros([x, y])
    index_m2 = np.zeros([x, y])
    index_n1 = 0
    index_n2 = 0

    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b1[ii-1, jj] != 1 and b1[ii+1, jj] != 1 and b1[ii, jj-1] != 1 and b1[ii, jj+1] != 1:
                index = False
            if b1[ii, jj] == 1 and index == True:
                min_index = index_n1 + 1
                if b1[ii - 1, jj] == 1:
                    min_index = np.min([min_index, index_m1[i - 1, j]])
                if b1[ii, jj - 1] == 1:
                    min_index = np.min([min_index, index_m1[i, j - 1]])
                k1 = 0
                while b1[ii, jj - k1] == 1:
                    index_m1[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b1[ii - k2, jj] == 1:
                    index_m1[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b1[ii, jj + k] == 1:
                    index_m1[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b1[ii + k, jj] == 1:
                    index_m1[i + k, j] = min_index
                    k = k + 1
                if b1[ii, jj - 1] != 1:
                    index_n1 = index_n1 + 1
                    k = 0
                    m = 0
                    while b1[ii, jj + k] == 1:
                        if b1[ii - 1, jj + k] == 1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n1 = index_n1 - 1

    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b2[ii - 1, jj] != -1 and b2[ii + 1, jj] != -1 and b2[ii, jj - 1] != -1 and b2[ii, jj + 1] != -1:
                index = False
            if b2[ii, jj] == -1 and index == True:
                min_index = index_n2 + 1
                if b2[ii - 1, jj] == -1:
                    min_index = np.min([min_index, index_m2[i - 1, j]])
                if b2[ii, jj - 1] == -1:
                    min_index = np.min([min_index, index_m2[i, j - 1]])
                k1 = 0
                while b2[ii, jj - k1] == -1:
                    index_m2[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b2[ii - k2, jj] == -1:
                    index_m2[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b2[ii, jj + k] == -1:
                    index_m2[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b2[ii + k, jj] == -1:
                    index_m2[i + k, j] = min_index
                    k = k + 1
                if b2[ii, jj - 1] != -1:
                    index_n2 = index_n2 + 1
                    k = 0
                    m = 0
                    while b2[ii, jj + k] == -1:
                        if b2[ii - 1, jj + k] == -1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n2 = index_n2 - 1

    return index_m1, index_n1, index_m2, index_n2


' a function for 1-sample & 1-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_1samp_1sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                             iter=1000):

    """
    1-sample & 1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        An expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1, which represent not significant point or significant
        point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_1samp(results[:, t], level, alternative='greater')
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        else:
            ps[t] = 0

    cluster_index, cluster_n = get_cluster_index_1d_1sided(ps)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t in range(x):
                if cluster_index[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t in range(x):
                    if cluster_index[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    return ps


' a function for 1-sample & 2-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_1samp_2sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                             iter=1000):

    """
    1-sample & 2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        An expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_1samp(results[:, t], level)
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        if p < p_threshold and ts[t] < 0:
            ps[t] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t in range(x):
                    if cluster_index1[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index1[t] == i + 1:
                        ps[t] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index2[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t in range(x):
                    if cluster_index2[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index2[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

        if newps[i + 1] == -1 and newps[i] != -1:
            index = 0
            while newps[i + 1 + index] == -1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    return ps


' a function for 1-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_1sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                       iter=1000):

    """
    1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results1 should be [n_subs, x]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results2 should be [n_subs, x]. n_subs represents the number of subjects. (Here, results1 >
        results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1, which represent not significant point or significant
        point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results1)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_rel(results1[:, t], results2[:, t], alternative='greater')
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        else:
            ps[t] = 0

    cluster_index, cluster_n = get_cluster_index_1d_1sided(ps)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t in range(x):
                if cluster_index[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t in range(x):
                    if cluster_index[t] == j + 1:
                        v = np.hstack((results1[:, t], results2[:, t]))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    return ps


' a function for 2-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                       iter=1000):

    """
    2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results1 should be [n_subs, x]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results2 should be [n_subs, x]. n_subs represents the number of subjects. (Here, results1 >
        results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results1)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_rel(results1[:, t], results2[:, t])
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        if p < p_threshold and ts[t] < 0:
            ps[t] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t in range(x):
                    if cluster_index1[t] == j + 1:
                        v = np.hstack((results1[:, t], results2[:, t]))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index1[t] == i + 1:
                        ps[t] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index2[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t in range(x):
                    if cluster_index2[t] == j + 1:
                        v = np.hstack((results1[:, t], results2[:, t]))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index2[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

        if newps[i + 1] == -1 and newps[i] != -1:
            index = 0
            while newps[i + 1 + index] == -1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    return ps


' a function for 1-sample & 1-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1samp_1sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=4,
                                             iter=1000):

    """
    1-sample & 1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    level : float. Default is 0.
        An expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 4.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1, which represent not significant point or
        significant point after cluster-based permutation test, respectively.
    """

    nsubs, x1, x2 = np.shape(results)

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_1samp(results[:, t1, t2], level, alternative='greater')
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            else:
                ps[t1, t2] = 0

    cluster_index, cluster_n = get_cluster_index_2d_1sided(ps)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    labels = label(ps, connectivity=1)
    nclusters = int(np.max(labels))
    for i in range(nclusters):
        n = (labels == i).sum()
        if n < n_threshold:
            for i1 in range(x1):
                for i2 in range(x2):
                    if labels[i1, i2] == i:
                        ps[i1, i2] = 0

    return ps


' a function for 1-sample & 2-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1samp_2sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=4,
                                             iter=1000):

    """
    1-sample & 2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    level : float. Default is 0.
        A expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 4.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """

    nsubs, x1, x2 = np.shape(results)

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_1samp(results[:, t1, t2], level)
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            if p < p_threshold and ts[t1, t2] < 0:
                ps[t1, t2] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_2d_2sided(ps)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index1[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test\n")
        print("Side 1 begin:")
        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index2[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("Side 2 begin:\n")
        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    labels = label(ps, connectivity=1)
    nclusters = int(np.max(labels))
    for i in range(nclusters):
        n = (labels == i).sum()
        if n < n_threshold:
            for i1 in range(x1):
                for i2 in range(x2):
                    if labels[i1, i2] == i:
                        ps[i1, i2] = 0

    return ps


' a function for 1-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=4,
                                       iter=1000):

    """
    1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results1 should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results2 should be [n_subs, x1, x2]. n_subs represents the number of subjects. (Here, results1 >
        results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 4.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1, which represent not significant point or
        significant point after cluster-based permutation test.
    """

    nsubs1, x11, x12 = np.shape(results1)
    nsubs2, x21, x22 = np.shape(results2)

    if nsubs1 != nsubs2 and x11 != x21 and x12 != x22:

        return "Invalid input!"

    nsubs = nsubs1
    x1 = x11
    x2 = x12

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_rel(results1[:, t1, t2], results2[:, t1, t2], alternative="greater")
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            else:
                ps[t1, t2] = 0

    cluster_index, cluster_n = get_cluster_index_2d_1sided(ps)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    labels = label(ps, connectivity=1)
    nclusters = int(np.max(labels))
    for i in range(nclusters):
        n = (labels == i).sum()
        if n < n_threshold:
            for i1 in range(x1):
                for i2 in range(x2):
                    if labels[i1, i2] == i:
                        ps[i1, i2] = 0

    return ps


' a function for 2-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=4,
                                       iter=1000):

    """
    2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results1 should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results2 should be [n_subs, x1, x2]. n_subs represents the number of subjects. (Here, results1 >
        results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 4.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test.
    """

    nsubs1, x11, x12 = np.shape(results1)
    nsubs2, x21, x22 = np.shape(results2)

    if nsubs1 != nsubs2 and x11 != x21 and x12 != x22:

        return "Invalid input!"

    nsubs = nsubs1
    x1 = x11
    x2 = x12

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_rel(results1[:, t1, t2], results2[:, t1, t2])
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            if p < p_threshold and ts[t1, t2] < 0:
                ps[t1, t2] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_2d_2sided(ps)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index1[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index2[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    labels = label(ps, connectivity=1)
    nclusters = int(np.max(labels))
    for i in range(nclusters):
        n = (labels == i).sum()
        if n < n_threshold:
            for i1 in range(x1):
                for i2 in range(x2):
                    if labels[i1, i2] == i:
                        ps[i1, i2] = 0

    return ps


' a function for smoothing the 1-D results'

def smooth_1d(x, n=5):

    """
    smoothing for 1-D results

    Parameters
    ----------
    x : array
        The results.
        The shape of x should be [n_sub, n_ts]. n_subs, n_ts represent the number of subjects and the number of
        time-points.
    n : int. Default is 5.
        The smoothing step is n.

    Returns
    -------
    x_smooth : array
        The results after smoothing.
        The shape of x_smooth should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and the number of
        time-points.
    """

    nsubs, nts = np.shape(x)

    x_smooth = np.zeros([nsubs, nts])

    ts1 = int(n / 2)
    ts2 = n - ts1

    for t in range(nts):

        if t >= ts1 and t < (nts - ts1):
            x_smooth[:, t] = np.average(x[:, t - ts1:t + ts2], axis=1)
        elif t < ts1:
            x_smooth[:, t] = np.average(x[:, :t + ts2], axis=1)
        else:
            x_smooth[:, t] = np.average(x[:, t - ts1:], axis=1)

    return x_smooth


' a function for smoothing the 2-D results'

def smooth_2d(x, n=5):

    """
    smoothing for 2-D results

    Parameters
    ----------
    x : array
        The results.
        The shape of x should be [n_sub, n_ts1, n_ts2]. n_subs represents the number of subjects. n_ts1 & n_ts2
        represent the numbers of time-points.
    n : int. Default is 5.
        The smoothing step is n.

    Returns
    -------
    x_smooth : array
        The results after smoothing.
        The shape of x should be [n_sub, n_ts1, n_ts2]. n_subs represents the number of subjects. n_ts1 & n_ts2
        represent the numbers of time-points.
    """

    nsubs, nts1, nts2 = np.shape(x)

    x_smooth = np.zeros([nsubs, nts1, nts2])

    ts1 = int(n / 2)
    ts2 = n - ts1

    for t1 in range(nts1):
        for t2 in range(nts2):

            if t1 < ts1 and t2 < ts1:
                x_smooth[:, t1, t2] = np.average(x[:, :t1 + ts2, :t2 + ts2], axis=(1, 2))
            elif t1 < ts1 and t2 >= ts1 and t2 < (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, :t1 + ts2, t2 - ts1:t2 + ts2], axis=(1, 2))
            elif t1 < ts1 and t2 >= (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, :t1 + ts2, t2 - ts1:], axis=(1, 2))
            elif t1 >= ts1 and t1 < (nts1 - ts1) and t2 < ts1:
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:t1 + ts2, :t2 + ts2], axis=(1, 2))
            elif t1 >= ts1 and t1 < (nts1 - ts1) and t2 >= ts1 and t2 < (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:t1 + ts2, t2 - ts1:t2 + ts2], axis=(1, 2))
            elif t1 >= ts1 and t1 < (nts1 - ts1) and t2 >= (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:t1 + ts2, t2 - ts1:], axis=(1, 2))
            elif t1 >= (nts1 - ts1) and t2 >= (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:, t2 - ts1:], axis=(1, 2))
            elif t1 >= (nts1 - ts1) and t2 >= ts1 and t2 < (nts2 - ts1):
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:, t2 - ts1:t2 + ts2], axis=(1, 2))
            elif t1 >= (nts1 - ts1) and t2 <= ts1:
                x_smooth[:, t1, t2] = np.average(x[:, t1 - ts1:, :t2 + ts2], axis=(1, 2))

    return x_smooth


' a function for showing the progress bar '

def show_progressbar(str, cur, total=100):

    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(str + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()