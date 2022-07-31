import os
import random
import numpy as np
import matplotlib.pyplot as plt

# locations of 4 centroid points
centroids = np.array([
    [0.2, 0.51],
    [0.08, 0.4],
    [0.32, 0.3],
    [0.8, 0.8]
])

print(centroids.shape)

# read data
plt.figure("Normalized clusters")
for dataset in range(1, 7):
    with open('../2 algorithm Data set/Q2/Dataset {}.txt'.format(dataset)) as f:
        lines = f.readlines()
    rawdata = np.zeros((np.shape(lines)[0], 2))
    npoints = rawdata.shape[0]    
    for i in range(np.shape(lines)[0]):
        tmp = lines[i].split()
        rawdata[i, 0] = float(tmp[0])
        rawdata[i, 1] = float(tmp[1])

    rawdata = (rawdata - np.min(rawdata, axis=0)) / (np.max(rawdata, axis=0) - np.min(rawdata, axis=0))
    plt.subplot(2, 3, dataset)
    plt.scatter(rawdata[:, 0], rawdata[:, 1], s=1, color='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, color='r')
    plt.axis([0, 1, 0, 1])

plt.show()

