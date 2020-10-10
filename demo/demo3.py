# -*- coding: utf-8 -*-

' a demo for comparing classification-based decoding and RSA '
# Users can learn how to use Neurora to decode information by RSA.

__author__ = 'Zitong Lu'

import os
import sys
import zipfile
import numpy as np
import scipy.io as sio
import h5py
from sklearn.svm import SVC
from neurora.stuff import permutation_test
from sklearn.metrics import accuracy_score
from six.moves import urllib
import matplotlib.pyplot as plt
from neurora.rdm_cal import eegRDM
from neurora.rsa_plot import plot_rdm, plot_tbytsim_withstats
from neurora.corr_cal_by_rdm import rdms_corr

url = 'https://attachment.zhaokuangshi.cn/BaeLuck_2018jn_data_ERP_5subs.zip'
filename = 'BaeLuck_2018jn_data_ERP_5subs.zip'
data_dir = 'data/'
classification_results_dir = 'classification_results/'
ctrsa_results_dir = 'rsa_results/'
filepath = data_dir + filename

"""******   Section 1: Download the data    ******"""

# Download the data

def show_progressbar(str, cur, total=100):

    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(str + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()

def schedule(blocknum,blocksize,totalsize):

    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0
    percent = percent * 100
    show_progressbar("Downloading", percent)

exist = os.path.exists(filepath)
if exist == False:
    os.makedirs(data_dir)
    urllib.request.urlretrieve(url, filepath, schedule)
    print('Download completes!')
elif exist == True:
    print('Data already exists!')

# unzip the data

def unzipfile(filepath, data_dir):

    with zipfile.ZipFile(filepath, 'r') as zip:
        zip.extractall(data_dir)
    print("Unzip completes!")

unzipfile(filepath, data_dir)



"""****** Section 2: Classification-based Decoding ******"""

# data preprocessing for classification-based decoding

# sub_ids
subs = ["201", "202", "203", "204", "205"]

exist = os.path.exists(data_dir + 'data_for_classification/ERP/')
if exist == False:
    os.makedirs(data_dir + 'data_for_classification/ERP/')

for sub in subs:
    data = sio.loadmat(data_dir + "data/ERP" + sub + ".mat")["filtData"][:, :, 250:]
    print(data.shape)
    # data.shape: n_trials, n_channels, n_times

    ori_label = np.loadtxt(data_dir + "labels/ori_" + sub + ".txt")[:, 1]
    pos_label = np.loadtxt(data_dir + "labels/pos_" + sub + ".txt")[:, 1]

    ori_subdata500 = np.zeros([16, 40, 27, 500], dtype=np.float)
    pos_subdata500 = np.zeros([16, 40, 27, 500], dtype=np.float)

    ori_labelindex = np.zeros([16], dtype=np.int)
    pos_labelindex = np.zeros([16], dtype=np.int)

    for i in range(640):
        label = int(ori_label[i])
        ori_subdata500[label, ori_labelindex[label]] = data[i]
        ori_labelindex[label] = ori_labelindex[label] + 1
        label = int(pos_label[i])
        pos_subdata500[label, pos_labelindex[label]] = data[i]
        pos_labelindex[label] = pos_labelindex[label] + 1

    ori_subdata = np.zeros([16, 40, 27, 100], dtype=np.float)
    pos_subdata = np.zeros([16, 40, 27, 100], dtype=np.float)

    for t in range(100):
        ori_subdata[:, :, :, t] = np.average(ori_subdata500[:, :, :, t * 5:t * 5 + 5], axis=3)
        pos_subdata[:, :, :, t] = np.average(pos_subdata500[:, :, :, t * 5:t * 5 + 5], axis=3)

    f = h5py.File(data_dir + "data_for_classification/ERP/" + sub + ".h5", "w")
    f.create_dataset("ori", data=ori_subdata)
    f.create_dataset("pos", data=pos_subdata)
    f.close()

# aftering the preprocessing above,
# we can obtain ERP data of orientation and position for each subject
# each subject's orientation ERP data's shape is [16, 40, 27, 100]
# 16: the number of conditions (here means 16 different orientation degrees)
# 40: the number of trials
# 27: the number of channels
# 100: the number of time-points (from -500 ms to 1500 ms, sample rate: 50 Hz)

# Linear-SVM decoding

exist = os.path.exists(classification_results_dir)
if exist == False:
    os.makedirs(classification_results_dir)

# orientation decoding
print("\nOrientation Decoding!")
subindex = 0
if os.path.exists(classification_results_dir + "ERP_ori.h5"):
    os.remove(classification_results_dir + "ERP_ori.h5")
f = h5py.File(classification_results_dir + "ERP_ori.h5", "w")
total = len(subs) * 10 * 3 * 100
for sub in subs:
    fdata = h5py.File(data_dir + "data_for_classification/ERP/" + sub + ".h5", "r")
    data = np.array(fdata["ori"])
    fdata.close()
    acc = np.zeros([10, 100, 3], dtype=np.float)
    for k in range(10):
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i * 13:i * 13 + 13], axis=1)
        y_train = np.zeros([2 * 16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i * 16 + j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2 * 16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, i] = accuracy_score(y_test, y_pred)
    subindex = subindex + 1
    f.create_dataset(sub, data=np.average(acc, axis=(0, 2)))
f.close()

# orientation decoding
print("\nPosition Decoding!")
subindex = 0
f = h5py.File(classification_results_dir + "ERP_pos.h5", "w")
total = len(subs) * 10 * 3 * 100
for sub in subs:
    fdata = h5py.File(data_dir + "data_for_classification/ERP/" + sub + ".h5", "r")
    data = np.array(fdata["pos"])
    fdata.close()
    acc = np.zeros([10, 100, 3], dtype=np.float)
    for k in range(10):
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i * 13:i * 13 + 13], axis=1)
        y_train = np.zeros([2 * 16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i * 16 + j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2 * 16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, i] = accuracy_score(y_test, y_pred)
    subindex = subindex + 1
    f.create_dataset(sub, data=np.average(acc, axis=(0, 2)))
f.close()



"""****** Section 3: Plot the classification-based decoding results ******"""

# plot the classification-based decoding results

# a function for plotting the time-by-time decoding results
def plot_tbytresults(decoding_results_dir, subs):
    f = h5py.File(decoding_results_dir, "r")
    nsubs = len(subs)
    rlts = np.zeros([nsubs, 100], dtype=np.float)
    subindex = 0
    for sub in subs:
        rlts[subindex] = np.array(f[sub])
        for t in range(100):
            if t <= 1:
                rlts[subindex, t] = np.average(rlts[subindex, :t + 3])
            if t > 1 and t < 98:
                rlts[subindex, t] = np.average(rlts[subindex, t - 2:t + 3])
            if t >= 98:
                rlts[subindex, t] = np.average(rlts[subindex, t - 2:])
        subindex = subindex + 1
    f.close()

    avg = np.average(rlts, axis=0)
    err = np.zeros([100], dtype=np.float)
    for t in range(100):
        err[t] = np.std(rlts[:, t], ddof=1) / np.sqrt(nsubs)

    ps = np.zeros([100], dtype=np.float)
    chance = np.full([16], 0.0625)
    for t in range(100):
        ps[t] = permutation_test(rlts[:, t], chance)
        if ps[t] < 0.05 and avg[t] > 0.0625:
            plt.plot(t * 0.02 - 0.5, 0.148, "s", color="orangered", alpha=0.8)
            xi = [t * 0.02 - 0.5, t * 0.02 + 0.02 - 0.5]
            ymin = [0.0625]
            ymax = [avg[t] - err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor="orangered", alpha=0.15)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0.0625))
    x = np.arange(-0.5 + 0.008, 1.5 + 0.008, 0.02)
    plt.fill_between(x, avg + err, avg - err, facecolor="orangered", alpha=0.8)
    plt.ylim(0.05, 0.15)
    plt.xlim(-0.5, 1.5)
    plt.xticks([-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    plt.tick_params(labelsize=12)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Classification Accuracy", fontsize=16)
    plt.show()

# plot orientation decoding results
print("Orientation Classification-based Decoding Results!")
plot_tbytresults(classification_results_dir + "ERP_ori.h5", subs)

# plot position decoding results
print("Position Classification-based Decoding Results!")
plot_tbytresults(classification_results_dir + "ERP_pos.h5", subs)



"""****** Section 4: RSA-based Decoding ******"""

# data preprocessing for classification-based decoding

if os.path.exists(data_dir + 'data_for_RSA/ERP/') == False:
    os.makedirs(data_dir + 'data_for_RSA/ERP/')

n = len(subs)
subindex = 0
for sub in subs:
    data = sio.loadmat(data_dir + "data/ERP" + sub + ".mat")["filtData"][:, :, 250:]
    # data.shape: n_trials, n_channels, n_times

    ori_label = np.loadtxt(data_dir + "labels/ori_" + sub + ".txt")[:, 1]
    pos_label = np.loadtxt(data_dir + "labels/pos_" + sub + ".txt")[:, 1]

    ori_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)
    pos_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)

    ori_labelindex = np.zeros([16], dtype=np.int)
    pos_labelindex = np.zeros([16], dtype=np.int)

    for i in range(640):
        label = int(ori_label[i])
        ori_subdata[label, ori_labelindex[label]] = data[i]
        ori_labelindex[label] = ori_labelindex[label] + 1
        label = int(pos_label[i])
        pos_subdata[label, pos_labelindex[label]] = data[i]
        pos_labelindex[label] = pos_labelindex[label] + 1

    f = h5py.File(data_dir + "data_for_RSA/ERP/" + sub + ".h5", "w")
    f.create_dataset("ori", data=ori_subdata)
    f.create_dataset("pos", data=pos_subdata)
    f.close()
    print(sub)

nsubs = len(subs)
data_ori_ERP = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)
data_pos_ERP = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)
subindex = 0
for sub in subs:
    print('Loading data of sub'+sub)
    f = h5py.File(data_dir+'data_for_RSA/ERP/'+sub+'.h5', 'r')
    ori_subdata = np.array(f['ori'])
    pos_subdata = np.array(f['pos'])
    f.close()
    data_ori_ERP[:, subindex] = ori_subdata
    data_pos_ERP[:, subindex] = pos_subdata
    subindex = subindex + 1

# calculate the RDMs

print("\nCalculate the Orientation RDMs!")
RDM_ori_ERP = eegRDM(data_ori_ERP, sub_opt=1, chl_opt=0, time_opt=1, time_win=5, time_step=5)
print("\nCalculate the Position RDMs!")
RDM_pos_ERP = eegRDM(data_pos_ERP, sub_opt=1, chl_opt=0, time_opt=1, time_win=5, time_step=5)
# shape of RDMs: [5, 100, 16, 16]

# establish a Coding RDM
model_RDM = np.zeros([16, 16], dtype=np.float)
for i in range(16):
    for j in range(16):
        diff = np.abs(i - j)
        if diff <= 8:
            model_RDM[i, j] = diff / 8
        else:
            model_RDM[i, j] = (16 - diff) / 8

conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]

# plot the Coding RDM
print("Coding RDM!")
plot_rdm(model_RDM, percentile=True, conditions=conditions)

# calculate the CTSimilarities between CTRDMs and Coding RDM
print("\nCalculate the Similarities of Orientation!")
Sim_ori_ERP = rdms_corr(model_RDM, RDM_ori_ERP)
print("\nCalculate the Similarities of Position!")
Sim_pos_ERP = rdms_corr(model_RDM, RDM_pos_ERP)

"""****** Section 5: Plot the RSA-based decoding results ******"""

# plot orientation decoding results
print("Orientation RSA-based Decoding Results!")
plot_tbytsim_withstats(Sim_ori_ERP, start_time=-0.5, end_time=1.5, color='orange', lim=[-0.1, 0.5])

# plot position decoding results
print("Position RSA-based Decoding Results!")
plot_tbytsim_withstats(Sim_pos_ERP, start_time=-0.5, end_time=1.5, color='orange', lim=[-0.1, 0.5])