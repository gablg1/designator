import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import svm

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from copy import deepcopy
#from ml_util import ml
from data import data
from data import image
import config
import core
from ml_util import ml
from cluster_recommender import ClusterRecommender
from duckling_recommender import DucklingRecommender
from random_recommender import RandomRecommender
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

amount = config.amount
ranks, names, histograms = data.getBinnedHistograms(amount, cut=True, big=False)

def test(data, recommender, fractionTrain=.8, highFactor=.1, verbose=False):
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
    m = xTrain.shape[0]

    train_colors, _, train_histograms = removeColors(xTrain, highFactor=highFactor)
    recommender.fit(train_histograms, train_colors)
    if verbose:
        print 'Done fitting'

    colors, quantities, histograms = removeColors(xTest, highFactor=highFactor)
    assert(colors.shape[0] == n)
    assert(histograms.shape[0] == n)
    numCorrect = 0

    D = histograms.shape[1]
    count = np.zeros(D)
    for color in colors:
        count[color] += 1

    tmp = count[np.where(count > 0)]
    color_mean = np.mean(tmp)
    color_stdev = np.std(tmp)
    print 'Color mean and stdev', color_mean, color_stdev

    recommendedColors = np.zeros((n))
    ignored = 0
    intersectionRatio = 0.
    for i in xrange(n):
        if i % 100 == 1:
            print 'Partial %d: %f' % (i, float(numCorrect) / (i - ignored + 1))

        color, amount = colors[i], quantities[i]
        # Ignore colors that might bias us
        if count[color] > color_mean + color_stdev:
            ignored += 1
            continue

        if verbose:
            print 'Testing site %s' % names[i]
            print 'Amount remmoved %d' % amount
        hist = histograms[i]

        intersectionRatio += core.clusterIntersectionRatio(hist, recommender.cluster(hist))
        #recommender.testClusters(hist)
        recommendedColor = recommender.predict(hist)
        r1, g1, b1 = image.binToRGB(color)
        r2, g2, b2 = image.binToRGB(recommendedColor)
        if verbose:
            print 'Removed color %d %d %d. Recommended color %d %d %d.' % (r1, g1, b1, r2, g2, b2)
            print 'Color distance: %d' % (image.binDistance(recommendedColor, color))
        recommendedColors[i] = recommendedColor

        if verbose:
            print 'Recommended color %d' % (recommendedColor)

        if recommendedColor == color:
            numCorrect += 1


    print 'Ignored: %d. Used: %d' % (ignored, n - ignored)
    print 'Mean cluster intersection ratio: %f' % (intersectionRatio / (n - ignored))
    print colorError(colors, recommendedColors)
    percentCorrect = float(numCorrect)/(n - ignored)
    return percentCorrect



def colorError(removed, recommended):
    n = len(removed)
    assert(len(recommended) == n)
    s = 0
    for i in xrange(n):
        s += image.binSquareDistance(removed[i], recommended[i])
    return float(s) / (n * 256 * 3)

def pickRandomColor(histogram):
    maxVal = np.amax(histogram)
    top_threshold = maxVal / 10.
    bottom_threshold = maxVal / 20.

    p = random.randint(0, len(histogram) - 1)
    while histogram[p] > top_threshold or histogram[p] < bottom_threshold:
        p = random.randint(0, len(histogram) - 1)
        top_threshold *= 1.001
        if top_threshold >= maxVal:
            top_threshold = maxVal
        bottom_threshold *= 0.999
        if bottom_threshold <= 0:
            bottom_threshold = 0
    return p

def pickColorToRemove(histogram, highFactor):
    prevDiff = 100000
    maxVal = np.amax(histogram)
    index = None
    for i in xrange(1, len(histogram)):
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
        color = pickRandomColor(bHistograms[i])
        colorsRemoved.append(color)
        quantityRemoved.append(bHistograms[i, color])
        ret[i, color] = 0
    return np.array(colorsRemoved), np.array(quantityRemoved), ret

if __name__ == '__main__':
    print 'Duckling Recommender'
    r = DucklingRecommender(cluster_size=15)
    print test(histograms, r, verbose=False)

    print 'Kmeans Classifier'
    r = ClusterRecommender(KMeans(n_clusters=15))
    print test(histograms, r, verbose=False)

    print 'Affinity Propagation Classifier'
    r = ClusterRecommender(AffinityPropagation(damping=0.8))
    print test(histograms, r, verbose=False)

    print 'Affinity Propagation Classifier'
    r = ClusterRecommender(AffinityPropagation(damping=0.99))
    print test(histograms, r, verbose=False)

    print 'Whole Set Classifier'
    r = ClusterRecommender(KMeans(n_clusters=1))
    print test(histograms, r, verbose=False)


    print 'Naive Bayes Classifier'
    print test(histograms, GaussianNB(), verbose=False)

    print 'Random Forest Classifier'
    print test(histograms, RandomForestClassifier())
#print 'Affinity Propagation Classifier'
#r = ClusterRecommender(AffinityPropagation(damping=0.7))
#print test(histograms, r, verbose=False)
