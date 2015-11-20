import numpy as np
# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from copy import deepcopy
#from ml_util import ml
from data import data
import recommend
import config
from ml_util import ml


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
    xTrain, xTest = ml.splitData(cluster, fractionTrain)
    prevDiff = 1000
    index = None
    copiedHists = np.array(xTest)
    testHists = []
    for testHist in xTest:
        maxVal = np.amax(testHist)
        for i in xrange(len(testHist)):
            ratio = np.fabs(testHist[i]/maxVal - highFactor)
            if ratio < prevDiff:
                index = i
        assert(index != None)
        print 'Removing color %d. Current amount is %d' % (index, testHist[index])
        testHist[index] = 0
        color, howMuch = recommend.recommendFromCluster(testHist, xTest)
        print 'Recommended color %d. Recommended amount is %d' % (color, howMuch)
        testHist[color]+= howMuch
        testHists.append(testHist)
    testHists = np.array(testHists)
    return ml.rmse(testHists, copiedHists)


amount = config.amount
ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)

print tester(histograms)
