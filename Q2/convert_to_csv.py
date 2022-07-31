import csv
import os
import numpy as np
from sys import argv

# read data
dataset = argv[1]
dataset_size = [34993, 19495, 25192, 26494, 23194, 17195]
with open('../2 algorithm Data set/Q2/Dataset {}.txt'.format(dataset)) as f:
    lines = f.readlines()
rawdata = np.zeros((np.shape(lines)[0], 2))
npoints = rawdata.shape[0]    
for i in range(np.shape(lines)[0]):
    tmp = lines[i].split()
    rawdata[i, 0] = float(tmp[0])
    rawdata[i, 1] = float(tmp[1])

rawdata = (rawdata - np.min(rawdata, axis=0)) / (np.max(rawdata, axis=0) - np.min(rawdata, axis=0))

# # random shuffle
# np.random.shuffle(rawdata)

# convert to tsv
if os.path.exists('./data/Dataset_{}.tsv'.format(dataset)):
    os.remove('./data/Dataset_{}.tsv'.format(dataset))
with open('./data/Dataset_{}.tsv'.format(dataset), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(dataset_size[int(dataset)-1]):
        tsv_writer.writerow([rawdata[i, 0], rawdata[i, 1], 1])