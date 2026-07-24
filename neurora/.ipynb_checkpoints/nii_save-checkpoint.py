# -*- coding: utf-8 -*-

' a module for saving the RSA results in a .nii file for fMRI '

__author__ = 'Zitong Lu'

import numpy as np
import nibabel as nib
from nilearn.image import smooth_img
import math
from neurora.stuff import fwe_correct, fdr_correct, cluster_fwe_correct, cluster_fdr_correct, get_HOcort, get_bg_ch2bet,\
    mask_to
from neurora.rsa_plot import plot_brainrsa_rlts

' a function for saving the searchlight correlation coefficients as a NIfTI file for fMRI '

def corr_save_nii(corrs, affine, filename=None, corr_mask=get_HOcort(), size=[60, 60, 60], ksize=[3, 3, 3],
                  strides=[1, 1, 1], p=1, r=0, correct_method=None, clusterp=0.05, smooth=True, plotrlt=True,
                  img_background=None):

    """
    Save the searchlight correlation coefficients as a NIfTI file for fMRI

    Parameters
    ----------
    corrs : array
        The similarities between behavioral data and fMRI data for searchlight.
        The shape of RDMs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for searchlight
        along the x, y, z axis and 2 represents a r-value and a p-value.
    affine : array or list
        The position information of the fMRI-image array data in a reference space.
    filename : string. Default is None - 'rsa_result.nii'.
        The file path+filename for the result .nii file.
        If the filename does not end in ".nii", it will be filled in automatically.
    corr_mask : string. Default is get_HOcort().
        The filename of a mask data for correcting the RSA result.
        It can just be one of your fMRI data files in your experiment for a mask file for ROI. If the corr_mask is a
        filename of a ROI mask file, only the RSA results in ROI will be visible.
    size : array or list [nx, ny, nz]. Default is [60, 60, 60].
        The size of the fMRI-img in your experiments.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    p : float. Default is 1.
        The threshold of p-values.
        Only the results those p-values are lower than this value will be visible.
    r : float. Default is 0.
        The threshold of r-values.
        Only the results those r-values are higher than this value will be visible.
    correct_method : None or string 'FWE' or 'FDR' or 'Cluster-FWE' or 'Cluster-FDR'. Default is None.
        The method for correcting the RSA results.
        If correct_method='FWE', here the FWE-correction will be used. If correct_methd='FDR', here the FDR-correction
        will be used. If correct_method='Cluster-FWE', here the Cluster-wise FWE-correction will be used. If
        correct_methd='Cluster-FDR', here the Cluster-wise FDR-correction will be used. If correct_method=None, no
        correction. If correct_method=None, no correction.
        Only when p<1, correct_method works.
    clusterp : float. Default is 0.05.
        The threshold of p-value for cluster-wise correction.
        Only when correct_method='Cluster-FDR' or 'Cluster-FWE', clusterp works.
    smooth : bool True or False. Default is True.
        Smooth the RSA result or not.
    plotrlt : bool True or False.
        Plot the RSA result automatically or not.
    img_background : None or string. Default if None.
        The filename of a background image that the RSA results will be plotted on the top of it.
        If img_background=None, the background will be ch2.nii.gz.
        Only when plotrlt=True, img_background works.

    Returns
    -------
    img : array
        The array of the correlation coefficients map.
        The shape is [nx, ny, nz]. nx, ny, nz represent the size of the fMRI-img.

    Notes
    -----
    A result .nii file of searchlight correlation coefficients will be generated at the corresponding address of
    filename.
    """

    if len(np.shape(corrs)) != 4 or len(np.shape(affine)) != 2 or np.shape(affine)[0] != 4 or np.shape(affine)[1] != 4:

        return "Invalid input!"

    # get the size of the fMRI-img
    nx = size[0]
    ny = size[1]
    nz = size[2]

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    rx = int((kx-1)/2)
    ry = int((ky-1)/2)
    rz = int((kz-1)/2)

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units in the x, y, z directions
    n_x = np.shape(corrs)[0]
    n_y = np.shape(corrs)[1]
    n_z = np.shape(corrs)[2]

    corrsr = corrs[:, :, :, 0]

    # initialize the img array to save the sum-r-value for each voxel
    img_nii = np.zeros([nx, ny, nz])

    # initialize a mask in order to record valid voxels (have qualified results)
    mask = np.zeros([nx, ny, nz], dtype=int)

    # get the p-values
    corrsp = corrs[:, :, :, 1]

    # do the correction
    if p < 1:

        # FDR-correction
        if correct_method == "FDR":
            corrsp = fdr_correct(corrsp, p_threshold=p)

        # FWE-correction
        if correct_method == "FWE":
            corrsp = fwe_correct(corrsp, p_threshold=p)

        # Cluster-wise FDR-correction
        if correct_method == "Cluster-FDR":
            corrsp = cluster_fdr_correct(corrsp, p_threshold1=p, p_threshold2=clusterp)

        # Cluster-wise FWE-correction
        if correct_method == "Cluster-FWE":
            corrsp = cluster_fwe_correct(corrsp, p_threshold1=p, p_threshold2=clusterp)

    # iterate through all the calculation units again

    # record the valid voxels
    # [n_x, n_y, n_z] expanses into [nx, ny, nz] based on ksize & strides
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                x = i * sx
                y = j * sy
                z = k * sz

                # p-values<threshold-p & r-values>threshold-r
                if (corrsp[i, j, k] < p) and (corrsr[i, j, k] > r):

                    mask[x + rx, y + ry, z + rz] = 1

                if (math.isnan(corrsr[i, j, k]) == False):

                    img_nii[x+rx, y+ry, z+rz] = img_nii[x+rx, y+ry, z+rz] + corrsr[i, j, k]

    # initialize the newimg array to calculate the avg-r-value for each voxel
    newimg_nii = np.full([nx, ny, nz], np.nan)

    # calculate the avg values of each valid voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # valid voxel
                if mask[i, j, k] == 1:
                    # sum-r-value/index
                    newimg_nii[i, j, k] = img_nii[i, j, k]


    # set filename for result .nii file
    if filename == None:
        filename = "rsa_result.nii"
    else:
        q = ".nii" in filename

        if q == True:
            filename = filename
        else:
            filename = filename+".nii"


    # corr_mask != None
    # use the mask file to correct RSA results
    # in order to avoid results showing outside of the brain
    if corr_mask == get_HOcort():

        mask_to(get_bg_ch2bet(), size, affine, filename=filename)
        mask = nib.load(filename).get_fdata()

    else:
        # load the array data of the mask file
        mask = nib.load(corr_mask).get_fdata()

    # do correction by the mask
    if corr_mask != None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (math.isnan(mask[i, j, k]) is True) or mask[i, j, k] == 0:
                        newimg_nii[i, j, k] = np.nan

    print(filename)

    print("Save RSA results.")

    # save the .nii file for RSA results
    file = nib.Nifti1Image(newimg_nii, affine)


    if smooth == True:
        # smooth the img data of the .nii file
        file = smooth_img(file, fwhm='fast')

    # save the result
    nib.save(file, filename)


    # determine if it has results
    norlt = np.isnan(newimg_nii).all()
    if norlt == True:
        print("No RSA results.")

    print("File("+filename+") saves successfully!")

    # determine plot the results or not
    if norlt == False and plotrlt == True:

        print("Plot RSA results.")
        plot_brainrsa_rlts(filename, background=img_background, type='r')

    return newimg_nii


' a function for saving the searchlight statistical results as a NIfTI file for fMRI '

def stats_save_nii(stats, affine, filename=None, corr_mask=get_HOcort(), size=[60, 60, 60], ksize=[3, 3, 3],
                   strides=[1, 1, 1], p=0.05, correct_method=None, clusterp=0.05, smooth=False, plotrlt=True,
                   img_background=None):

    """
    Save the searchlight RSA statistical results as a NIfTI file for fMRI

    Parameters
    ----------
    stats : array
        The statistical results between behavioral data and fMRI data for searchlight.
        The shape of RDMs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units for searchlight
        along the x, y, z axis and 2 represents a t-value and a p-value.
        If the filename does not end in ".nii", it will be filled in automatically.
    affine : array or list
        The position information of the fMRI-image array data in a reference space.
    filename : string. Default is None - 'rsa_result.nii'.
        The file path+filename for the result .nii file.
    corr_mask : string
        The filename of a mask data for correcting the RSA result.
        It can just be one of your fMRI data files in your experiment for a mask file for ROI. If the corr_mask is a
        filename of a ROI mask file, only the RSA results in ROI will be visible.
    size : array or list [nx, ny, nz]. Default is [60, 60, 60].
        The size of the fMRI-img in your experiments.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    p : float. Default is 0.05.
        The threshold of p-values.
        Only the results those p-values are lower than this value will be visible.
    correct_method : None or string 'FWE' or 'FDR' or 'Cluster-FWE' or 'Cluster-FDR'. Default is None.
        The method for correcting the RSA results.
        If correct_method='FWE', here the FWE-correction will be used. If correct_methd='FDR', here the FDR-correction
        will be used. If correct_method='Cluster-FWE', here the Cluster-wise FWE-correction will be used. If
        correct_methd='Cluster-FDR', here the Cluster-wise FDR-correction will be used. If correct_method=None, no
        correction.
        Only when p<1, correct_method works.
    clusterp : float. Default is 0.05.
        The threshold of p-value for cluster-wise correction.
        Only when correct_method='Cluster-FDR' or 'Cluster-FWE', clusterp works.
    smooth : bool True or False.  Default is False.
        Smooth the RSA result or not.
    plotrlt : bool True or False.  Default is True.
        Plot the RSA result automatically or not.
    img_background : None or string. Default if None.
        The filename of a background image that the RSA results will be plotted on the top of it.
        If img_background=None, the background will be ch2.nii.gz.
        Only when plotrlt=True, img_background works.

    Returns
    -------
    img : array
        The array of the statistical results t-values map.
        The shape is [nx, ny, nz]. nx, ny, nz represent the size of the fMRI-img.

    Notes
    -----
    A result .nii file of searchlight statistical results will be generated at the corresponding address of filename.
    """


    if len(np.shape(stats)) != 4 or len(np.shape(affine)) != 2 or np.shape(affine)[0] != 4 or np.shape(affine)[1] != 4:

        return "Invalid input!"

    # get the size of the fMRI-img
    nx = size[0]
    ny = size[1]
    nz = size[2]

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    rx = int((kx-1)/2)
    ry = int((ky-1)/2)
    rz = int((kz-1)/2)

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units in the x, y, z directions
    n_x = np.shape(stats)[0]
    n_y = np.shape(stats)[1]
    n_z = np.shape(stats)[2]

    img_nii = np.zeros([nx, ny, nz])

    # initialize a mask in order to record valid voxels (have qualified results)
    mask = np.zeros([nx, ny, nz], dtype=int)

    # get the p-values
    statsp = stats[:, :, :, 1]
    statst = stats[:, :, :, 0]

    # calculate the number of voxels for correction
    fadeimg = np.zeros([nx, ny, nz], dtype=int)

    # iterate through all the calculation units

    # calculate the indexs
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                x = i*sx
                y = j*sy
                z = k*sz

                if statsp[i, j, k] < 1:
                    img_nii[x + rx, y + ry, z + rz] = statst[i, j, k]
                if statsp[i, j, k] < p:
                    fadeimg[x + rx, y + ry, z + rz] = 1

    n_corrected = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if fadeimg[i, j, k] == 1:
                    n_corrected = n_corrected + 1

    print(str(n_corrected)+" voxels will be corrected.")

    # do the correction
    if p < 1:

        # FDR-correction
        if correct_method == "FDR":
            statsp = fdr_correct(statsp, p_threshold=p)

        # FWE-correction
        if correct_method == "FWE":
            statsp = fwe_correct(statsp, p_threshold=p)

        # Cluster-wise FDR-correction
        if correct_method == "Cluster-FDR":
            statsp = cluster_fdr_correct(statsp, p_threshold1=p, p_threshold2=clusterp)

        # Cluster-wise FWE-correction
        if correct_method == "Cluster-FWE":
            statsp = cluster_fwe_correct(statsp, p_threshold1=p, p_threshold2=clusterp)

    # iterate through all the calculation units again

    print("Record the valid voxels.")

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                x = i * sx
                y = j * sy
                z = k * sz

                if statsp[i, j, k] < p:
                    mask[x + rx, y + ry, z + rz] = 1

    # initialize the newimg array to calculate the avg-r-value for each voxel
    newimg_nii = np.full([nx, ny, nz], np.nan)

    # set filename for result .nii file
    if filename == None:
        filename = "rsa_result.nii"
    else:
        q = ".nii" in filename

        if q == True:
            filename = filename
        else:
            filename = filename + ".nii"

    # corr_mask != None
    # use the mask file to correct RSA results
    # in order to avoid results showing outside of the brain
    if corr_mask == get_HOcort():

        mask_to(get_bg_ch2bet(), size, affine, filename)
        cmask = nib.load(filename).get_fdata()

    else:
        # load the array data of the mask file
        cmask = nib.load(corr_mask).get_fdata()

    # calculate the avg values of each valid voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # valid voxel
                if (math.isnan(cmask[i, j, k]) == False) and cmask[i, j, k] != 0 and mask[i, j, k] == 1:
                    # sum-r-value/index
                    newimg_nii[i, j, k] = img_nii[i, j, k]

    print("Get RSA results.")

    print(filename)

    print("Save RSA results.")

    # save the .nii file for RSA results
    file = nib.Nifti1Image(newimg_nii, affine)


    if smooth == True:
        print("Smooth the results.")
        # smooth the img data of the .nii file
        file = smooth_img(file, fwhm='fast')

    # save the result
    nib.save(file, filename)

    # determine if it has results
    norlt = np.isnan(newimg_nii).all()
    if norlt == True:
        print("No RSA results.")

    print("File("+filename+") saves successfully!")

    # determine plot the results or not
    if norlt == False and plotrlt == True:

        print("Plot RSA results.")
        plot_brainrsa_rlts(filename, background=img_background, type='t')

    return newimg_nii
