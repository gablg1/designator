import numpy as np
from sklearn.naive_bayes import GaussianNB
# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from copy import deepcopy
#from ml_util import ml
from data import data
from recommender import Recommender
import config
from ml_util import ml

amount = config.amount
ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)

def tester(data, recommender, fractionTrain=.5, highFactor=.1):
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
    xTrain, xTest = ml.splitData(data, fractionTrain)
    n = xTest.shape[0]
    colors, histograms = removeColors(xTest, highFactor=highFactor)
    assert(len(colors) == n)
    assert(histograms.shape[0] == n)
    numCorrect = 0

    recommender.train(histograms, colors)

    for i in xrange(n):
        colors, histograms = removeColors(xTest, highFactor=highFactor)
    	color, amount = colors[i]
    	print 'Testing site %s' % names[i]
        print 'Removed color %d. Amount removed: %d' % (color, amount)
        hist = histograms[i]
        elem, recommendedColor = recommender.recommendFromCluster(hist, xTrain)

        print 'Recommended color %d' % (recommendedColor)
        print 'Recommended from website %s' % names[elem]
        if recommendedColor == color:
            numCorrect += 1
    return float(numCorrect)/float(n)

def pickColorToRemove(histogram, highFactor):
    prevDiff = 100000
    maxVal = np.amax(histogram)
    index = None
    for i in xrange(len(histogram)):
        ratio = np.fabs(histogram[i]/maxVal - highFactor)
        if ratio < prevDiff:
            index = i
            prevDiff = ratio
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


# Try to recommend colors using GaussianNB
#gnb = GaussianNB()


print tester(histograms, Recommender())

