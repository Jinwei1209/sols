import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from utils import *
from sys import argv

# read data
with open('../2 algorithm Data set/Q1/Q1 Dataset 1.txt') as f:
    lines = f.readlines()
rawdata = np.zeros((np.shape(lines)[0]-2, 2))
npoints = rawdata.shape[0]    
for i in range(np.shape(lines)[0]-2):
    tmp = lines[i+2].split()
    rawdata[i, 0] = float(tmp[0])
    rawdata[i, 1] = float(tmp[1])

# calculate first order derivatives "d"
peakgroup = 100
SlopeThreshold = 0.0012  # 0.0012 to miss the smallest peak / 0.001 to pick all peaks
smoothwidth = 50

d = derivative(rawdata[:, 1])

plt.figure("First order derivative of raw data")
plt.plot(rawdata[:, 0], d)

# smooth first order derivatives
d = smooth(d, smoothwidth=smoothwidth)
d = smooth(d, smoothwidth=smoothwidth)

plt.figure("Smoothed first order derivative of raw data")
plt.plot(rawdata[:, 0], d)
plt.axhline(y=0.0, color='r')

n = int(round(peakgroup/2+1))
Peaks = []
for j in range(2*int(round(smoothwidth/2))-2, npoints-smoothwidth):
    # Detects zero-crossing
    if (np.sign(d[j]) > np.sign(d[j+1])):
        # if slope of derivative is larger than SlopeThreshold 
        if (d[j]-d[j+1] > SlopeThreshold):
            xx = np.zeros(peakgroup)
            yy = np.zeros(peakgroup)
            # Create sub-group of points near peak
            for k in range(0, peakgroup):
                groupindex = j+k-n+2;
                if (groupindex < 1): 
                    groupindex = 1
                if (groupindex > npoints): 
                    groupindex = npoints
                xx[k] = rawdata[groupindex, 0]
                yy[k] = rawdata[groupindex, 1]
            Peaks.append(j)

# plt.figure("Raw data")
# plt.plot(rawdata[:, 0], rawdata[:, 1])
# plt.plot(rawdata[Peaks, 0], rawdata[Peaks, 1], '*', color='red', markersize=2)

# calculate heights accurately (but slower)
if int(argv[1]) == 1:
    # detect all non-peak regions
    nonPeaks1 = []
    peakwidth = 50
    ampThreshold = 0.02
    for j in range(0, npoints):
        flag = [j in range(i-peakwidth, i+peakwidth) for i in Peaks]
        if (sum(flag) == 0) and (abs(d[j] < ampThreshold)):
            nonPeaks1.append(j)

    # reverse rawdata and detect again
    d = derivative(rawdata[::-1, 1])
    d = smooth(d, smoothwidth=smoothwidth)
    d = smooth(d, smoothwidth=smoothwidth)
    d = d[::-1]
    nonPeaks2 = []
    for j in range(0, npoints):
        flag = [j in range(i-peakwidth, i+peakwidth) for i in Peaks]
        if (sum(flag) == 0) and (abs(d[j] < ampThreshold)):
            nonPeaks2.append(j)

    # merge two list without duplication
    nonPeaks = [value for value in nonPeaks1 if value in nonPeaks2]

    Peaks_indicator = np.ones(npoints)
    Peaks_indicator[nonPeaks] = 0
    plt.figure("nonPeak regions")
    plt.plot(rawdata[:, 0], rawdata[:, 1])
    plt.plot(rawdata[:, 0], Peaks_indicator*50, color='red')

    # moving average of baselines
    nonPeaks_value = rawdata[nonPeaks, 1]
    nonPeaks_value = smooth(nonPeaks_value, smoothwidth=500)
    baselines = np.zeros(npoints)
    baselines[nonPeaks] = nonPeaks_value

    # calculate heights of each peak
    heights = []
    for j in Peaks:
        for i in range(0, len(nonPeaks)):
            if j > nonPeaks[i] and j < nonPeaks[i+1]:
                heights.append(rawdata[j, 1] - (baselines[nonPeaks[i]] + baselines[nonPeaks[i+1]])/2)

    # draw heights
    plt.figure("Final results with heights and positions")
    plt.plot(rawdata[:, 0], rawdata[:, 1], linewidth=1)
    plt.plot(rawdata[Peaks, 0], rawdata[Peaks, 1], '*', color='red', markersize=2)
    plt.plot(nonPeaks, nonPeaks_value, color='red', linewidth=1)
    for i in range(len(Peaks)):
        point1 = [rawdata[Peaks[i], 0], rawdata[Peaks[i], 1]-heights[i]]
        point2 = [rawdata[Peaks[i], 0], rawdata[Peaks[i], 1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, ':', color='red', linewidth=1)
    plt.show()
    plt.savefig('results_heights_with_baseline.png', dpi=200)

    # write results into .csv file 
    dict = {'Positions': Peaks, 'Heights': heights} 
    df = pd.DataFrame(dict)
    df.to_csv('results_heights_with_baseline.csv')

# calculate heights fast (but not accurate)
elif int(argv[1]) == 2:
    peakwidth = 200
    Peak_intervals = []
    Peak_intervals.append([Peaks[0]-peakwidth, Peaks[0]+peakwidth])
    idx = 0
    for i in Peaks[1:]:
        if (i-peakwidth < Peak_intervals[idx][1]):
            Peak_intervals[idx][1] = i + peakwidth
        else:
            Peak_intervals.append([i-peakwidth, i+peakwidth])
            idx += 1

    # calculate heights of each peak
    heights = []
    for j in Peaks:
        for i in range(0, len(Peak_intervals)):
            if j > Peak_intervals[i][0] and j < Peak_intervals[i][1]:
                heights.append(rawdata[j, 1] - (rawdata[Peak_intervals[i][0], 1] + rawdata[Peak_intervals[i][1], 1])/2)

    # draw heights
    plt.figure("Final results with heights and positions")
    plt.plot(rawdata[:, 0], rawdata[:, 1], linewidth=1)
    plt.plot(rawdata[Peaks, 0], rawdata[Peaks, 1], '*', color='red', markersize=2)
    for i in range(len(Peaks)):
        point1 = [rawdata[Peaks[i], 0], rawdata[Peaks[i], 1]-heights[i]]
        point2 = [rawdata[Peaks[i], 0], rawdata[Peaks[i], 1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, ':', color='red', linewidth=1)
    plt.show()
    plt.savefig('results_coarse_heights.png', dpi=200)

    # write results into .csv file 
    dict = {'Positions': Peaks, 'Heights': heights} 
    df = pd.DataFrame(dict)
    df.to_csv('results_coarse_heights.csv')
    