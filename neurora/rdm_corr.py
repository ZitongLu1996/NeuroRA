# -*- coding: utf-8 -*-

' a module for calculating the correlation coefficient between two RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau

' a function for calculating the Spearman correlation coefficient between two RDMs '
def rdm_correlation_spearman(RDM1, RDM2, rescale=False):


    cons = np.shape(RDM1)[0]

    print(cons)

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    print(np.shape(RDM1))
    print(np.shape(RDM2))

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        for i in range(cons):
            for j in range(cons):
                if i != j:
                    RDM1[i, j] = float((RDM1[i, j]-minvalue)/(maxvalue-minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        for i in range(cons):
            for j in range(cons):
                if i != j:
                    RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons-1):

        for j in range(cons-1-i):

            v1[nn] = RDM1[i, i+j+1]
            v2[nn] = RDM2[i, i+j+1]

            nn = nn + 1

    print(v1)
    print(v2)

    return spearmanr(v1, v2)

' a function for calculating the Pearson correlation coefficient between two RDMs '
def rdm_correlation_pearson(RDM1, RDM2, rescale=False):

    cons = np.shape(RDM1)[0]

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons - 1):

        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]

            nn = nn + 1

    return pearsonr(v1, v2)

' a function for calculating the Kendalls tau correlation coefficient between two RDMs '
def rdm_correlation_kendall(RDM1, RDM2, rescale=False):

    cons = np.shape(RDM1)[0]

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons - 1):

        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]

            nn = nn + 1

    return kendalltau(v1, v2)

' a function for calculating the Cosine Similarity between two RDMs '

def rdm_similarity(RDM1, RDM2, rescale=False):

    cons = np.shape(RDM1)[0]

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons - 1):

        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]

            nn = nn + 1

    V1 = np.mat(v1)
    V2 = np.mat(v2)

    num = float(V1 * V2.T)

    denom = np.linalg.norm(V1) * np.linalg.norm(V2)

    cos = num / denom

    similarity = 0.5 + 0.5 * cos

    return similarity

' a fuction for calculating the Euclidean Distances between two RDMs '

def rdm_distance(RDM1, RDM2, rescale=False):
    cons = np.shape(RDM1)[0]

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons - 1):

        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]

            nn = nn + 1

    dist = np.linalg.norm(v1 - v2)

    return dist

' a function for permutation test between two RDMs '

def rdm_permutation(RDM1, RDM2, iter=1000, rescale=False):
    cons = np.shape(RDM1)[0]

    n = 0

    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    nn = 0

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        vrdm = np.reshape(RDM1, [cons*cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        vrdm = np.reshape(RDM2, [cons * cons])
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]
        if maxvalue != minvalue:
            for i in range(cons):
                for j in range(cons):
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    for i in range(cons - 1):

        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]

            nn = nn + 1

    diff = abs(np.average(v1) - np.average(v2))

    v = np.hstack((v1, v2))

    print(v)

    nv = v.shape[0]

    ni = 0

    for i in range(iter):
        vshuffle = np.random.permutation(v)

        vshuffle1 = vshuffle[:int(nv/2)]
        vshuffle2 = vshuffle[int(nv/2):]

        diff_i = abs(np.average(vshuffle1) - np.average(vshuffle2))

        if diff_i >= diff:
            ni = ni + 1

    p = ni/iter

    return p
