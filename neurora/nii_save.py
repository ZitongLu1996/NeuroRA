# -*- coding: utf-8 -*-

' a module for saving the RSA results in a .nii file for fMRI '

__author__ = 'Zitong Lu'

import os
import numpy as np
import nibabel as nib
from nilearn.image import smooth_img
from neurora.stuff import fwe_correct, fdr_correct, correct_by_threshold,\
    get_HOcort, get_bg_ch2bet, mask_to
from neurora.rsa_plot import plot_brainrsa_rlts

np.seterr(divide="ignore", invalid="ignore")


def _validate_searchlight_results(results):
    """
    Validate searchlight result array.

    Parameters
    ----------
    results : array_like, shape [n_x, n_y, n_z, 2]
        Searchlight results.

        The last dimension contains:

        - results[..., 0] : value map, e.g., r-values or t-values
        - results[..., 1] : p-values

    Returns
    -------
    results : ndarray, shape [n_x, n_y, n_z, 2]
        Validated searchlight results.
    """

    results = np.asarray(results, dtype=float)

    if results.ndim != 4 or results.shape[-1] != 2:
        raise ValueError(
            "results should have shape [n_x, n_y, n_z, 2]. "
            "The last dimension should contain [value, p]."
        )

    return results


def _validate_affine(affine):
    """
    Validate affine matrix.

    Parameters
    ----------
    affine : array_like, shape [4, 4]
        NIfTI affine matrix.

    Returns
    -------
    affine : ndarray, shape [4, 4]
        Validated affine matrix.
    """

    affine = np.asarray(affine, dtype=float)

    if affine.shape != (4, 4):
        raise ValueError("affine should have shape [4, 4].")

    return affine


def _validate_size(size):
    """
    Validate NIfTI image size.

    Parameters
    ----------
    size : tuple or list, length 3
        Image size.

    Returns
    -------
    size : tuple of int
        Validated image size.
    """

    if len(size) != 3:
        raise ValueError("size should have length 3.")

    size = tuple(map(int, size))

    if min(size) < 1:
        raise ValueError("size should contain positive integers.")

    return size


def _validate_ksize_strides(ksize, strides):
    """
    Validate searchlight kernel size and strides.

    Parameters
    ----------
    ksize : tuple or list, length 3
        Searchlight kernel size.

    strides : tuple or list, length 3
        Searchlight strides.

    Returns
    -------
    ksize : tuple of int
        Validated kernel size.

    strides : tuple of int
        Validated strides.
    """

    if len(ksize) != 3 or len(strides) != 3:
        raise ValueError("ksize and strides should both have length 3.")

    ksize = tuple(map(int, ksize))
    strides = tuple(map(int, strides))

    if min(ksize) < 1:
        raise ValueError("ksize should contain positive integers.")

    if min(strides) < 1:
        raise ValueError("strides should contain positive integers.")

    return ksize, strides


def _validate_correct_method(correct_method):
    """
    Validate correction method.

    Parameters
    ----------
    correct_method : {None, "FWE", "FDR"}
        Multiple-comparison correction method.

    Returns
    -------
    correct_method : None or str
        Validated correction method.
    """

    if correct_method is None:
        return None

    if not isinstance(correct_method, str):
        raise TypeError("correct_method should be None, 'FWE', or 'FDR'.")

    correct_method = correct_method.upper()

    if correct_method not in {"FWE", "FDR"}:
        raise ValueError(
            "correct_method should be None, 'FWE', or 'FDR'. "
            "Cluster-FWE and Cluster-FDR are no longer supported."
        )

    return correct_method


def _validate_plot_type(plot_type):
    """
    Validate plot type.

    Parameters
    ----------
    plot_type : {"r", "t"}
        Plot type.

    Returns
    -------
    plot_type : str
        Validated plot type.
    """

    if plot_type not in {"r", "t"}:
        raise ValueError("plot_type should be 'r' or 't'.")

    return plot_type


def _format_nii_filename(filename, default="rsa_result.nii"):
    """
    Format output NIfTI filename.

    Parameters
    ----------
    filename : str or None
        Output filename.

    default : str, default="rsa_result.nii"
        Default filename when filename is None.

    Returns
    -------
    filename : str
        Formatted filename.
    """

    if filename is None:
        return default

    if not isinstance(filename, str) or filename == "":
        raise ValueError("filename should be a non-empty string or None.")

    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return filename

    return filename + ".nii"


def _apply_p_correction(p_values, p_threshold, correct_method):
    """
    Apply p-value correction to a 3-D p-value map.

    Parameters
    ----------
    p_values : ndarray, shape [n_x, n_y, n_z]
        P-value map.

    p_threshold : float
        P-value threshold.

    correct_method : {None, "FWE", "FDR"}
        Correction method.

    Returns
    -------
    corrected_p : ndarray, shape [n_x, n_y, n_z]
        Corrected p-value map.
    """

    if correct_method is None or p_threshold >= 1:
        return p_values

    if correct_method == "FWE":
        return fwe_correct(p_values, p_threshold=p_threshold)

    if correct_method == "FDR":
        return fdr_correct(p_values, p_threshold=p_threshold)

    raise ValueError("correct_method should be None, 'FWE', or 'FDR'.")


def _searchlight_to_volume(values, p_values, size, ksize, strides, p_threshold=1,
                           value_threshold=None):
    """
    Project searchlight values back to a 3-D NIfTI volume.

    Parameters
    ----------
    values : ndarray, shape [n_x, n_y, n_z]
        Searchlight value map, such as r-values or t-values.

    p_values : ndarray, shape [n_x, n_y, n_z]
        Searchlight p-value map.

    size : tuple of int, length 3
        Target image size [nx, ny, nz].

    ksize : tuple of int, length 3
        Searchlight kernel size.

    strides : tuple of int, length 3
        Searchlight strides.

    p_threshold : float, default=1
        P-value threshold.

    value_threshold : float or None, default=None
        Optional value threshold. If None, no value-thresholding is applied.

    Returns
    -------
    volume : ndarray, shape [nx, ny, nz]
        Thresholded 3-D volume.
    """

    values = np.asarray(values, dtype=float)
    p_values = np.asarray(p_values, dtype=float)

    if values.shape != p_values.shape:
        raise ValueError("values and p_values should have the same shape.")

    nx, ny, nz = size
    kx, ky, kz = ksize
    sx, sy, sz = strides

    rx = int((kx - 1) / 2)
    ry = int((ky - 1) / 2)
    rz = int((kz - 1) / 2)

    n_x, n_y, n_z = values.shape

    volume = np.full((nx, ny, nz), np.nan, dtype=float)

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                x = i * sx + rx
                y = j * sy + ry
                z = k * sz + rz

                if x < 0 or x >= nx or y < 0 or y >= ny or z < 0 or z >= nz:
                    continue

                value = values[i, j, k]
                p_value = p_values[i, j, k]

                if not (np.isfinite(value) and np.isfinite(p_value)):
                    continue

                if p_value >= p_threshold:
                    continue

                if value_threshold is not None and value <= value_threshold:
                    continue

                volume[x, y, z] = value

    return volume


def _load_or_create_mask(corr_mask, size, affine):
    """
    Load or create a correction mask.

    Parameters
    ----------
    corr_mask : str or None
        Mask filename.

    size : tuple of int
        Target image size.

    affine : ndarray, shape [4, 4]
        Target affine.

    Returns
    -------
    mask_data : ndarray or None
        Mask data. If corr_mask is None, returns None.
    """

    if corr_mask is None:
        return None

    if corr_mask == get_HOcort():
        temp_mask_filename = "_neurora_temp_mask.nii"
        mask_to(get_bg_ch2bet(), size, affine, filename=temp_mask_filename)
        mask_data = nib.load(temp_mask_filename).get_fdata()

        try:
            os.remove(temp_mask_filename)
        except OSError:
            pass

        return mask_data

    mask_data = nib.load(corr_mask).get_fdata()

    return mask_data


def _apply_mask(volume, mask_data):
    """
    Apply a 3-D mask to a 3-D volume.

    Parameters
    ----------
    volume : ndarray, shape [nx, ny, nz]
        Input volume.

    mask_data : ndarray or None
        Mask data.

    Returns
    -------
    masked_volume : ndarray, shape [nx, ny, nz]
        Masked volume.
    """

    if mask_data is None:
        return volume

    mask_data = np.asarray(mask_data, dtype=float)

    if mask_data.shape != volume.shape:
        raise ValueError("mask_data should have the same shape as the output volume.")

    masked_volume = np.array(volume, copy=True)

    valid_mask = (mask_data != 0) & np.isfinite(mask_data)
    masked_volume[~valid_mask] = np.nan

    return masked_volume


def _save_and_optionally_plot(volume, affine, filename, smooth=True, plotrlt=True,
                              img_background=None, plot_type="r", colorbar_label=None):
    """
    Save a 3-D volume as a NIfTI file and optionally plot it.

    Parameters
    ----------
    volume : ndarray, shape [nx, ny, nz]
        Output volume.

    affine : ndarray, shape [4, 4]
        NIfTI affine.

    filename : str
        Output filename.

    smooth : bool, default=True
        Whether to smooth the output image.

    plotrlt : bool, default=True
        Whether to plot the result automatically.

    img_background : str or None, default=None
        Background image filename.

    plot_type : {"r", "t"}, default="r"
        Plot type passed to plot_brainrsa_rlts.

    colorbar_label : str or None, default=None
        Colorbar label. Currently reserved for future plot_brainrsa_rlts support.

    Returns
    -------
    saved_volume : ndarray
        The saved volume before smoothing.
    """

    if not isinstance(smooth, bool):
        raise TypeError("smooth should be True or False.")

    if not isinstance(plotrlt, bool):
        raise TypeError("plotrlt should be True or False.")

    img = nib.Nifti1Image(volume, affine)

    if smooth:
        print("Smooth the results.")
        img = smooth_img(img, fwhm="fast")

    nib.save(img, filename)

    no_result = np.isnan(volume).all()

    if no_result:
        print("No RSA results.")

    print(f"File({filename}) saves successfully!")

    if (not no_result) and plotrlt:
        print("Plot RSA results.")

        # colorbar_label is reserved here.
        # If plot_brainrsa_rlts later supports it, pass it there.
        plot_brainrsa_rlts(
            filename,
            background=img_background,
            type=plot_type,
        )

    return volume


def save_nii(results, affine, filename=None, corr_mask=get_HOcort(), size=(60, 60, 60),
             ksize=(3, 3, 3), strides=(1, 1, 1), p=0.05, value_threshold=None,
             correct_method=None, nvoxels_threshold=None, smooth=True, plotrlt=True,
             img_background=None, plot_type="r", colorbar_label=None):
    """
    Save fMRI RSA searchlight results as a NIfTI file.

    Parameters
    ----------
    results : array_like, shape [n_x, n_y, n_z, 2]
        Searchlight results.

        The last dimension contains:

        - results[..., 0] : value map, such as r-values or t-values
        - results[..., 1] : p-values

    affine : array_like, shape [4, 4]
        NIfTI affine matrix.

    filename : str or None, default=None
        Output NIfTI filename.

        If None, "rsa_result.nii" will be used.

    corr_mask : str or None, default=get_HOcort()
        Mask filename.

        If None, no mask will be applied.

    size : tuple or list of int, default=(60, 60, 60)
        Target image size [nx, ny, nz].

    ksize : tuple or list of int, default=(3, 3, 3)
        Searchlight kernel size [kx, ky, kz].

    strides : tuple or list of int, default=(1, 1, 1)
        Searchlight strides [sx, sy, sz].

    p : float, default=0.05
        P-value threshold.

    value_threshold : float or None, default=None
        Optional threshold for values.

        For correlation maps, this can be used as an r-value threshold.
        For statistical maps, this is usually None.

    correct_method : {None, "FWE", "FDR"}, default=None
        Multiple-comparison correction method.

        - None : no p-value correction
        - "FWE" : Bonferroni FWE correction
        - "FDR" : Benjamini-Hochberg FDR correction

    nvoxels_threshold : int or None, default=None
        Minimum cluster size in voxels.

        If not None, small clusters in the final volume will be removed using
        correct_by_threshold().

    smooth : bool, default=True
        Whether to smooth the output NIfTI image.

    plotrlt : bool, default=True
        Whether to plot the result automatically.

    img_background : str or None, default=None
        Background image filename used for plotting.

    plot_type : {"r", "t"}, default="r"
        Plot type passed to plot_brainrsa_rlts.

    colorbar_label : str or None, default=None
        Colorbar label. Reserved for future plot_brainrsa_rlts support.

    Returns
    -------
    volume : ndarray, shape [nx, ny, nz]
        Saved result volume before smoothing.
    """

    results = _validate_searchlight_results(results)
    affine = _validate_affine(affine)
    size = _validate_size(size)
    ksize, strides = _validate_ksize_strides(ksize, strides)
    correct_method = _validate_correct_method(correct_method)
    plot_type = _validate_plot_type(plot_type)

    if not isinstance(p, (int, float)):
        raise TypeError("p should be a number.")

    p = float(p)

    if p <= 0 or p > 1:
        raise ValueError("p should be in the interval (0, 1].")

    if value_threshold is not None and not isinstance(value_threshold, (int, float)):
        raise TypeError("value_threshold should be a number or None.")

    if nvoxels_threshold is not None:
        if not isinstance(nvoxels_threshold, int) or nvoxels_threshold < 1:
            raise ValueError("nvoxels_threshold should be a positive integer or None.")

    filename = _format_nii_filename(filename, default="rsa_result.nii")

    values = results[..., 0]
    p_values = results[..., 1]

    p_values = _apply_p_correction(
        p_values=p_values,
        p_threshold=p,
        correct_method=correct_method,
    )

    volume = _searchlight_to_volume(
        values=values,
        p_values=p_values,
        size=size,
        ksize=ksize,
        strides=strides,
        p_threshold=p,
        value_threshold=value_threshold,
    )

    if nvoxels_threshold is not None:
        volume = correct_by_threshold(
            volume,
            threshold=nvoxels_threshold,
            connectivity=1,
        )

    mask_data = _load_or_create_mask(
        corr_mask=corr_mask,
        size=size,
        affine=affine,
    )

    volume = _apply_mask(
        volume=volume,
        mask_data=mask_data,
    )

    print(filename)
    print("Save RSA results.")

    volume = _save_and_optionally_plot(
        volume=volume,
        affine=affine,
        filename=filename,
        smooth=smooth,
        plotrlt=plotrlt,
        img_background=img_background,
        plot_type=plot_type,
        colorbar_label=colorbar_label,
    )

    return volume