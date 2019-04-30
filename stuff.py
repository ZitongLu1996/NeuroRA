# -*- coding: utf-8 -*-

' a module for some simple but important processes '

__author__ = 'Zitong Lu'

import nibabel as nib

def limtozero(x):
    if x < 1e-15:
        x = 0
    return x

def get_affine(file_name):

    img = nib.load(file_name)

    return img.affine