import numpy as np

def derivative(y):
    '''
    First derivative of vector using 2-point central difference.
    '''
    npoints = y.shape[0]
    results = np.zeros(npoints)
    results[0] = y[1] - y[0]
    results[npoints-1] = y[npoints-1] - y[npoints-2]
    for j in range(1, npoints-1):
        results[j] = (y[j+1] - y[j-1])/2
    return results

def smooth(y, smoothwidth):
    '''
    Moving average smoothing of vector y
    '''
    w = int(round(smoothwidth))
    SumPoints = sum(y[0:w])
    s = np.zeros(y.shape[0])
    halfw = int(round(w/2))
    L = y.shape[0]
    for k in range(1, L-w):
        s[k+halfw-1] = SumPoints
        SumPoints = SumPoints - y[k]
        SumPoints = SumPoints + y[k+w]
    s[k+halfw] = sum(y[L-w+1:L+1])
    s = s / w
    return s