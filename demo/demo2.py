# -*- coding: utf-8 -*-

' a demo based on Haxby fMRI data '
# Users can learn how to use Neurora to do research based on fMRI data.

__author__ = 'Zitong Lu'

from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import numpy as np
import pandas as pd
import nibabel as nib
from neurora.stuff import get_affine, datamask
from neurora.nps_cal import nps_fmri, nps_fmri_roi
from neurora.rsa_plot import plot_rdm
from neurora.rdm_cal import fmriRDM_roi, fmriRDM
from neurora.corr_to_nii import corr_save_nii

"""**********       Section 1: loading example data        **********"""
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
    np.savetxt("demo02/data"+str(i+1)+".txt", np.reshape(fmri_data[i], [nx*ny*nz]))

# get fmri data under 'face'-condition
face_img = nib.Nifti1Image(fmri_data[0], affine=img.affine)
# have a look
plotting.plot_epi(face_img)
plotting.show()

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

print(nps_roi)



"""**********       Section 4: Calculating the neural pattern similarity (Searchlight)        **********"""

# calculate the neural pattern similarity (NPS) between two stimulus
nps = nps_fmri(nps_fmri_data)

# convert the NPS results into a .nii file
savefilename = "nps_img"
affine = get_affine(mask_filename)
corr_save_nii(nps, filename=savefilename, affine=affine, size=[nx, ny, nz], plotrlt=False)

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
plot_rdm(rdm_roi, rescale=True, conditions=categories)



"""**********       Section 5: Calculating the RDM by Searchlight and Plotting        **********"""

# calculate the RDMs by Searchlight
fmri_RDMs = fmriRDM(fmri_data)

# plot one of the RDMs
print(fmri_RDMs[20, 30, 30])
plot_rdm(fmri_RDMs[20, 30, 30], rescale=True, conditions=categories)