# -*- coding: utf-8 -*-

' a module for plotting the RDM '

__author__ = 'Zitong Lu'

import numpy as np
import matplotlib.pyplot as plt

def plot_rdm_1(rdm):

    if len(np.shape(rdm)) != 2:

        return None

    a, b = np.shape(rdm)

    if a != b:

        return None

    plt.imshow(rdm, cmap=plt.cm.jet, clim=(0, 1))

    plt.axis("off")

    plt.colorbar()

    plt.show()

def plot_rdm_2(rdm):

    if len(np.shape(rdm)) != 2:

        return None

    a, b = np.shape(rdm)

    if a != b:

        return None

    plt.imshow(rdm, cmap=plt.cm.Greens, clim=(0, 1))

    plt.axis("off")

    for i in range(a):
        for j in range(b):
            text = plt.text(i, j, float('%.4f'%rdm[i, j]),
                           ha="center", va="center", color="blue")

    plt.colorbar()

    plt.show()