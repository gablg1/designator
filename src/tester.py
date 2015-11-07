import numpy as np
# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from copy import deepcopy
from data import data
import recommend


def tester(cluster, fractionTrain=.9, highFactor=.1):
    """
    Parameters:
        cluster: an array of arrays
        fractionTrain: a float specifying percent of data to use as train
        highFactor: ratio of any given element to the largest element in the array
    Returns:
        the RMSE produced by removing a specific color, then adding it back

    Algorithm:
        Works by first finding the max value in the histogram and then trying to find the
        index of the histogram that contains the value that is closest (in terms of a ration
        with the max value) to highFactor. Removes this color, then passes it to recommender
        to get back a value which it then adds to the testHistogram, then takes the rmse between
        the original and the modified
    """
    endTrain = round(cluster.shape[0] * fractionTrain)
    xTrain = cluster[0:endTrain,]
    xTest = cluster[endTrain:,]
    mse = 0.
    prevDiff = 1000
    index = None
    for testHist in xTest:
        maxVal = np.amax(testHist)
        copiedHist = deepcopy(testHist)
        for i in xrange(len(testHist)):
            ratio = np.fabs(testHist[i]/maxVal - highFactor)
            if ratio < prevDiff:
                index = i
        assert(index != None)
        testHist[i] = 0
        color, howMuch = recommend.recommendFromCluster(testHist, xTrain)
        testHist[color]+= howMuch
        diff = copiedHist - testHist
        mse += np.sum(diff**2)
    rmse = np.sqrt(mse/len(xTest))
    return rmse


amount = 'top-15k'
ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)
small_histograms = histograms[200:]

print tester(small_histograms)
