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
        to get back a value which it then adds to the histogramogram, then takes the rmse between
        the original and the modified
    """
    xTrain, xTest = ml.splitData(cluster, fractionTrain)
    index = None
    copiedHists = np.array(xTest)

    n = xTest.shape[0]
    colors, histograms = removeColors(xTest, highFactor=highFactor)
    assert(len(colors) == n)
    assert(histograms.shape[0] == n)

    for i in xrange(n):
    	color, amount = colors[i]
    	print 'Histogram #%d' % i
        print 'Removed color %d. Amount removed: %d' % (color, amount)
        hist = histograms[i]
        recommendedColor, howMuch = recommend.recommendFromCluster(hist, xTrain)
        print 'Recommended color %d. Recommended amount: %d' % (recommendedColor, howMuch)
        histograms[i, color] += howMuch
    return ml.rmse(histograms, xTest)

def pickColorToRemove(histogram, highFactor):
    prevDiff = 1000
    maxVal = np.amax(histogram)
    for i in xrange(len(histogram)):
        ratio = np.fabs(histogram[i]/maxVal - highFactor)
        if ratio < prevDiff:
            index = i
    assert(index != None)
    return index

def removeColors(bHistograms, highFactor):
    N = bHistograms.shape[0]
    ret = np.copy(bHistograms)
    colors = []
    for i in xrange(N):
    	color = pickColorToRemove(bHistograms[i], highFactor=highFactor)
    	colors.append((color, bHistograms[i, color]))
    	ret[i, color] = 0
    return colors, ret



amount = config.amount
ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)

print tester(histograms)
