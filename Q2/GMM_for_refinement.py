import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sys import argv
from sklearn import mixture

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
with open('./temp/Assignment.txt'.format(dataset)) as f:
    lines = f.readlines()
predictions = np.zeros(np.shape(lines)[0], int)
npoints = predictions.shape[0]    
for i in range(np.shape(lines)[0]):
    tmp = lines[i].split()
    predictions[i] = float(tmp[0])

rawdata = rawdata[:predictions.shape[0], :]

# colors
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']),
                                    int(max(predictions) + 1))))
colors = np.append(colors, ["#000000"])

# select the 2nd cluster for GMM
rawdata_2nd = rawdata[predictions==1]
np.random.seed(1)
gmm = mixture.GaussianMixture(n_components=1)
gmm.fit(rawdata_2nd)
predictions_2nd = np.ones(np.shape(rawdata_2nd)[0], dtype=np.int8) * 1

# plot the contour of GMM fitting
print(gmm.means_)
print('\n')
print(gmm.covariances_)
X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape((50,50))
plt.contour(X, Y, Z)
# plt.scatter(rawdata_2nd[:, 0], rawdata_2nd[:, 1], s=1, color=colors[predictions_2nd])
# plt.show()
# plt.savefig('cluster_2nd_GMM_{}.png'.format(dataset), dpi=200)

# filter outliers
thrd = 2
Z = gmm.score_samples(rawdata_2nd)
predictions_2nd[Z < thrd] = -1
plt.scatter(rawdata_2nd[:, 0], rawdata_2nd[:, 1], s=1, color=colors[predictions_2nd])
plt.savefig('./results/cluster_2nd_GMM_{}.png'.format(dataset), dpi=200)

# save new labels with outliers removed
predictions[predictions==1] = predictions_2nd
with open('./temp/Assignment_wo_outliers.txt', 'w') as f:
    for label in predictions:
        f.write(str(label))
        f.write('\n')

# plot clusters with outliers
plt.figure()
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
plt.savefig('./results/cluster_with_outliers_{}.png'.format(dataset), dpi=200)