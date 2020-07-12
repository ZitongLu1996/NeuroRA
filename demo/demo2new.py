# -*- coding: utf-8 -*-

' a demo based on Haxby fMRI data '
# Users can learn how to use Neurora to do research based on fMRI data.

__author__ = 'Zitong Lu'

from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import numpy as np
import pandas as pd
import nibabel as nib
from neurora.stuff import get_affine, datamask, get_bg_ch2bet, correct_by_threshold
from neurora.nps_cal import nps_fmri, nps_fmri_roi
from neurora.rsa_plot import plot_rdm
from neurora.rdm_cal import fmriRDM_roi, fmriRDM
from neurora.corr_cal_by_rdm import fmrirdms_corr
from neurora.nii_save import corr_save_nii
import matplotlib.pyplot as plt

def plot_rdm(rdm, rescale=False, lim=[0, 1], conditions=None, con_fontsize=12, cmap=None, filename="test"):

    """
    Plot the RDM

    Parameters
    ----------
    rdm : array or list [n_cons, n_cons]
        A representational dissimilarity matrix.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.
    conditions : string-array or string-list. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'jet'.
    """

    # get the number of conditions
    cons = rdm.shape[0]

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of RDM cannot be 2*2. Here NeuroRA cannot plot this RDM.")

        return None

    # determine if it's a square
    a, b = np.shape(rdm)
    if a != b:
        return None

    # rescale the RDM
    if rescale == True:

        # flatten the RDM
        vrdm = np.reshape(rdm, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        rdm[i, j] = float((rdm[i, j] - minvalue) / (maxvalue - minvalue))

    # plot the RDM
    min = lim[0]
    max = lim[1]
    if cmap == None:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
    else:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}

    if rescale == True:
        cb.set_label("Dissimilarity (Rescaling)", fontdict=font)
    elif rescale == False:
        cb.set_label("Dissimilarity", fontdict=font)

    if conditions != None:
        print("1")
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.savefig(filename+".png")

    plt.show()

def plot_brainrsa_montage(img, threshold=None, slice=[6, 6, 6], background=get_bg_ch2bet(), type='r', filename="test"):

    """
    Plot the RSA-result by different cuts

    Parameters
    ----------
    img : string
        The file path of the .nii file of the RSA results.
    threshold : None or int. Default is None.
        The threshold of the number of voxels used in correction.
        If threshold=n, only the similarity clusters consisting more than threshold voxels will be visible. If it is
        None, the threshold-correction will not work.
    slice : array
        The point where the cut is performed.
        If slice=[slice_x, slice_y, slice_z], slice_x, slice_y, slice_z represent the coordinates of each cut in the x,
        y, z direction. If slice=[[slice_x1, slice_x2], [slice_y1, slice_y2], [slice_z1, slice_z2]], slice_x1 & slice_x2
        represent the coordinates of each cut in the x direction, slice_y1 & slice_y2 represent the coordinates of each
        cut in the y direction, slice_z1 & slice_z2 represent the coordinates of each cut in the z direction.
    background : Niimg-like object or string. Default is stuff.get_bg_ch2bet()
        The background image that the RSA results will be plotted on top of.
    type : string 'r' or 't'
        The type of result (r-values or t-values).
    """

    imgarray = nib.load(img).get_data()

    if (imgarray == np.nan).all() == True:

        print("No Valid Results")

    else:

        if threshold != None:
            imgarray = nib.load(img).get_data()
            affine = get_affine(img)
            imgarray = correct_by_threshold(imgarray, threshold)
            img = nib.Nifti1Image(imgarray, affine)

        slice_x = slice[0]
        slice_y = slice[1]
        slice_z = slice[2]

        if type == 'r':
            vmax = 1
        if type == 't':
            vmax = 7

        if slice_x != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='x', cut_coords=slice_x,
                                title="Similarity -sagittal", draw_cross=True, vmax=vmax)

        if slice_y != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='y', cut_coords=slice_y,
                                title="Similarity -coronal", draw_cross=True, vmax=vmax)

        if slice_z != 0:
            plotting.plot_stat_map(stat_map_img=img, bg_img=background, display_mode='z', cut_coords=slice_z,
                                title="Similarity -axial", draw_cross=True, vmax=vmax)

        plt.savefig(filename+".png")

        plt.show()

"""**********       Section 1: Loading example data        **********"""
""" Here, we use Nilearn toolbox for loading data and processing """
""" you can learn this process from Nilearn (http://nilearn.github.io/index.html) """

# load Haxby dataset (here, we only use subject2's data for this example)
haxby_dataset = datasets.fetch_haxby()

# load the fMRI data filename & mask data filename
func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask

# read label information of the experiment
labelinfo = pd.read_csv(haxby_dataset.session_target[0], sep=' ')
labels = labelinfo['labels']



"""**********       Section 2: Preprocessing        **********"""

# get mask data NumPy array
maskdata = nib.load(mask_filename).get_data()

# get the size of the data
nx, ny, nz = maskdata.shape

# labels of seven ategories
categories = ["face", "cat", "house", "chair", "shoe", "bottle", "scissors"]
# numbe of conidtions: 7
ncon = len(categories)

# get fmri data under 7 conditions
# here we average the data under different conditions
fmri_data = np.full([ncon, nx, ny, nz], np.nan)

for i in range(ncon):
    img = mean_img(index_img(func_filename, labels.isin([categories[i]])))
    fmri_data[i] = datamask(img.get_data(), maskdata)

# get fmri data under 'face'-condition
face_img = nib.Nifti1Image(fmri_data[0], affine=img.affine)
# have a look
plotting.plot_epi(face_img)
plotting.show()

# reshaoe the data: [ncon, nx, ny, nz] -> [ncon, nsubs, nx, ny, nz]
# here just one subject's data
fmri_data = np.reshape(fmri_data, [ncon, 1, nx, ny, nz])



"""**********       Section 3: Calculating the neural pattern similarity (for ROI)        **********"""

# get mask of 'mask_face' in the dataset
mask_face_filename = haxby_dataset.mask_face[0]
mask_face_data = nib.load(mask_face_filename).get_data()

# get input data under two condition
# here, "face"-condition vs. "cat"-condition
nps_fmri_data = fmri_data[[0, 6]]

# calculate the neural pattern similarity (NPS) for ROI between two stimulus
nps_roi = nps_fmri_roi(nps_fmri_data, mask_face_data)

# print the NPS result
print(nps_roi)



"""**********       Section 4: Calculating the neural pattern similarity (Searchlight)        **********"""

# calculate the neural pattern similarity (NPS) between two stimulus
nps = nps_fmri(nps_fmri_data)

# convert the NPS results into a .nii file
savefilename = "demo2_nps_img"
affine = get_affine(mask_filename)
corr_save_nii(nps[0], filename=savefilename, affine=affine, size=[nx, ny, nz], smooth=False, plotrlt=False)

# have a look
plotting.plot_epi(savefilename+".nii")
plotting.show()



"""**********       Section 5: Calculating the RDM for ROI and Plotting        **********"""

# get mask of "mask_vt" in the dataset
mask_vt_filename = haxby_dataset.mask_face[0]
mask_vt_data = nib.load(mask_vt_filename).get_data()

# calculate the RDM for ROI
rdm_roi = fmriRDM_roi(fmri_data, mask_vt_data)

# plot the RDM
plot_rdm(rdm_roi, rescale=True, conditions=categories, filename="demo2-s5")



"""**********       Section 6: Calculating the RDM by Searchlight and Plotting        **********"""

# calculate the RDMs by Searchlight
fmri_RDMs = fmriRDM(fmri_data)

# plot one of the RDMs
plot_rdm(fmri_RDMs[20, 30, 30], rescale=True, conditions=categories, filename="demo2-s6")



"""**********       Section 7: Calculating the representational similarities       **********"""
"""**********             between a coding model and neural activities           **********"""

# Create a RDM for "animate-inanimate" coding model
# which means the representations of animate matters are highly similar
# and the representations of inanimate matters are highly similar
model_RDM = np.array([[0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0]])

# plot the model RDM
plot_rdm(model_RDM, conditions=categories, filename="demo2-s7")

# calculate the similarities between model RDM and searchlight RDMs
corrs = fmrirdms_corr(model_RDM, fmri_RDMs)



"""**********       Section 8: Saving the RSA result and Plotting       **********"""

# load the filename of anatomical image as the background for plotting
ant_filename = haxby_dataset.anat[0]

# get the affine info
affine = get_affine(mask_filename)

# save the RSA result as a .nii file
# and visualize the result automatically
# p < 0.05, FDR-correct
rsarltfilename = "demo2_rsarlt_img.nii"
img = corr_save_nii(corrs, filename=rsarltfilename, affine=affine, corr_mask=mask_filename, size=[40, 64, 64], p=0.001, plotrlt=True, img_background=ant_filename)

# Users can plot the RSA results independently by functions below
from neurora.rsa_plot import plot_brainrsa_regions
from neurora.rsa_plot import plot_brainrsa_montage
from neurora.rsa_plot import plot_brainrsa_glass
from neurora.rsa_plot import plot_brainrsa_surface

# here use a [5, 5, 5] cube to remove the significant area smaller than it
# before filtering
plot_brainrsa_montage(rsarltfilename, slice=[[-25], 0, 0], background=ant_filename, filename="demo2-s8-1")
# after filtering
plot_brainrsa_montage(rsarltfilename, threshold=125, slice=[[-25], 0, 0], background=ant_filename, filename="demo2-s8-2")
