Overview
========

Representational Similarity Analysis (RSA) has become a popular and effective method to measure the representation of multivariable neural activity in different modes.

NeuroRA is an easy-to-use toolbox based on Python, which can do some works about RSA among nearly all kinds of neural data, including behavioral, EEG, MEG, fNIRS, sEEG, ECoG, fMRI and some other neuroelectrophysiological data. In addition, users can do Neural Pattern Similarity (NPS), Spatiotemporal Pattern Similarity (STPS) & Inter-Subject Correlation (ISC) on NeuroRA.

NeuroRA includes some typical features below:

* Calculate the Neural Pattern Similarity (NPS)
* Calculate the Spatiotemporal Neural Pattern Similarity (STPS)
* Calculate the Inter-Subject Correlation (ISC)
* Calculate the Representational Dissimilarity Matrix (RDM)
* Calculate the Representational Similarity based on RDMs
* One-Step Realize Representational Similarity Analysis (RSA)
* Statistical Analysis
* Save the RSA result as a NIfTI file for fMRI
* Visualization for RSA results

Required Dependencies:

* `Numpy <http://www.numpy.org>`_: a fundamental package for scientific computing.
* `SciPy <https://www.scipy.org/scipylib/index.html>`_: a package that provides many user-friendly and efficient numerical routines.
* `Matplotlib <https://matplotlib.org>`_: a Python 2D plotting library.
* `NiBabel <https://nipy.org/nibabel/>`_: a package prividing read +/- write access to some common medical and neuroimaging file formats.
* `Nilearn <https://nilearn.github.io/>`_: a Python module for fast and easy statistical learning on NeuroImaging data.
* `MNE-Python <https://mne.tools/>`_: a Python software for exploring, visualizing, and analyzing human neurophysiological data.