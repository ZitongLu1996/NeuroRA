<meta name="google-site-verification" content="nSJP12yLCr6zLi5RzrkcJOfIrlut0Ro3Y63OOZ0_hdU" />
# NeuroRA

![ ](img/logo.jpg)

**A Python Toolbox of Representational Analysis from Multimode Neural Data**

## Overview
**Representational Similarity Analysis (RSA)** has become a popular and effective method to measure the representation of multivariable neural activity in different modes.

**NeuroRA** is an easy-to-use toolbox based on **Python**, which can do some works about RSA among nearly all kinds of neural data, including **behavioral, EEG, MEG, fNIRS, ECoG, electrophysiological and fMRI data**.

## Installation
> pip install NeuroRA 

## Documentation & How to Use
You can read or download the [Tutorial here](https://zitonglu1996.github.io/NeuroRA/neurora/Tutorial.pdf).

## Required Dependencies:
- **[Numpy](http://www.numpy.org)**: a fundamental package for scientific computing
- **[Matplotlib](https://matplotlib.org)**: a Python 2D plotting library
- **[NiBabel](https://nipy.org/nibabel/)**: a package prividing read +/- write access to some common medical and neuroimaging file formats
- **[Nilearn](https://nilearn.github.io/)**: a Python module for fast and easy statistical learning on NeuroImaging data
- **[MNE-Python](https://mne.tools/)**: a Python software for exploring, visualizing, and analyzing human neurophysiological data

## Features

- Calculate the Neural Pattern Similarity (NPS)

- Calculate the Representational Dissimilarity Matrix (RDM)

    > for each subject / for each channel / for each time-point / searchlight / for ROI / all in

- Calculate the Representational Similarity based on RDMs

    > for each subject / for each channel / for each time-point / searchlight / for ROI / all in

- One-Step Realize Representational Similarity Analysis (RSA)

    > for each subject / for each channel / for each time-point / searchlight / for ROI / all in

- Save the RSA result as a NIfTI file for fMRI

- Visualization for RSA results

## Typical Visualization Demos

- Representational Dissimilarity Matrix

    ![ ](img/01.png)

- Representational Similarities by time sequence

    ![ ](img/02.png)

- RSA-result for fMRI

    ![ ](img/03.png)
    ![ ](img/04.png)

## Paper

Lu, Z., & Ku, Y. NeuroRA: A Python toolbox of representational analysis from multi-modal neural data. (bioRxiv: https://doi.org/10.1101/2020.03.25.008086)

## About NeuroRA

**Noteworthily**, this toolbox is currently only a **test version**. 
If you have any question, find some bugs or have some useful suggestions while using, you can email me and I will be happy and thankful to know.
>My email address: 
>zitonglu1996@gmail.com

>My personal homepage:
>https://zitonglu1996.github.io

