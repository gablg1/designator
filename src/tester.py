import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from copy import deepcopy
#from ml_util import ml
from data import data
import config
from ml_util import ml
from cluster_recommender import ClusterRecommender

amount = config.amount
ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)

def tester(data, recommender, fractionTrain=.5, highFactor=.1, verbose=False):
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

    train_colors, _, train_histograms = removeColors(xTrain, highFactor=highFactor)
    recommender.fit(train_histograms, train_colors)
    if verbose:
    	print 'Done fitting'

    colors, quantities, histograms = removeColors(xTest, highFactor=highFactor)
    assert(colors.shape[0] == n)
    assert(histograms.shape[0] == n)
    numCorrect = 0


    for i in xrange(n):
    	color, amount = colors[i], quantities[i]
        if verbose:
            print 'Testing site %s' % names[i]
            print 'Removed color %d. Amount removed: %d' % (color, amount)
        hist = histograms[i]

        # This is used for cluster recommendations
        #elem, recommendedColor = recommender.recommendFromCluster(hist, xTrain)
        #if verbose:
        #   print 'Recommended from website %s' % names[elem]

        # This is used for vanilla classifiers
        recommendedColor = recommender.predict(hist)

        if verbose:
            print 'Recommended color %d' % (recommendedColor)

        if recommendedColor == color:
            numCorrect += 1
    return float(numCorrect)/n

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
    colorsRemoved = []
    quantityRemoved = []
    for i in xrange(N):
    	color = pickColorToRemove(bHistograms[i], highFactor=highFactor)
    	colorsRemoved.append(color)
    	quantityRemoved.append(bHistograms[i, color])
    	ret[i, color] = 0
    return np.array(colorsRemoved), np.array(quantityRemoved), ret


# Try to recommend colors using GaussianNB
#gnb = GaussianNB()

print 'Naive Bayes Classifier'
print tester(histograms, GaussianNB(), verbose=False)
print 'Random Forest Classifier'
print tester(histograms, RandomForestClassifier())
print 'Kmeans Classifier'
r = ClusterRecommender(KMeans(n_clusters=5))
print tester(histograms, r, verbose=True)

