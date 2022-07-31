import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sys import argv

# read data
dataset = argv[1]
with open('./data/Dataset_{}.tsv'.format(dataset)) as f:
    lines = f.readlines()
rawdata = np.zeros((np.shape(lines)[0], 2)) 
for i in range(np.shape(lines)[0]):
    tmp = lines[i].split()
    rawdata[i, 0] = float(tmp[0])
    rawdata[i, 1] = float(tmp[1])

# read clustering result
with open('./temp/Assignment.txt') as f:
    lines = f.readlines()
predictions = np.zeros(np.shape(lines)[0], int)
npoints = predictions.shape[0]    
for i in range(np.shape(lines)[0]):
    tmp = lines[i].split()
    predictions[i] = float(tmp[0])

rawdata = rawdata[:predictions.shape[0], :]

# plot clustering results
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']),
                                    int(max(predictions) + 1))))
# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])
plt.scatter(rawdata[:, 0], rawdata[:, 1], s=1, color=colors[predictions])

# read clustering centroids index
with open('./temp/Centroid.txt') as f:
    lines = f.readlines()
centroids_index = np.zeros(np.shape(lines)[0], int)
for i in range(np.shape(lines)[0]):
    tmp = lines[i].split()
    centroids_index[i] = float(tmp[0])

# plot clustering centroids
plt.scatter(rawdata[centroids_index, 0], rawdata[centroids_index, 1], s=20, color='r')
plt.axis([0, 1, 0, 1])
plt.show()
plt.savefig('cluster_{}.png'.format(dataset), dpi=200)
