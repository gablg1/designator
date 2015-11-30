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
from ml_util import ml
from cluster_recommender import ClusterRecommender
from random_recommender import RandomRecommender
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

amount = config.amount
ranks, names, histograms = data.getBinnedHistograms(amount, cut=True, big=False)
SAMPLE_SIZE = 10

def tester(data, recommender, fractionTrain=.5, highFactor=.1, verbose=False, plot=False):
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
    xTest = xTest[:SAMPLE_SIZE, :]

    n, D = xTest.shape
    assert(D == image.BINNED_DIM)
    m = xTrain.shape[0]

    train_names = names[:m]

    train_colors, _, train_histograms = removeColors(xTrain, highFactor=highFactor)
    recommender.fitWithPlot(train_histograms, train_colors, train_names)
    if verbose:
    	print 'Done fitting'

    colors, quantities, histograms = removeColors(xTest, highFactor=highFactor)
    assert(colors.shape[0] == n)
    assert(histograms.shape[0] == n)
    numCorrect = 0
    if plot:
        colorRecommend = []
        namesRecommend = []
        colorRemoved = []
        clusterLoc= []
        #clusterIndexList = []

    D = histograms.shape[1]
    count = np.zeros(D)
    for color in colors:
        count[color] += 1

    clusters = []
    recommendedColors = np.zeros((n))
    ignored = 0
    for i in xrange(n):
        if i % 100 == 1:
            print 'Partial %d: %f' % (i, float(numCorrect) / i)

    	color, amount = colors[i], quantities[i]
        # Ignore colors that might bias us
        if count[color] > 10:
        	ignored += 1
        	continue

        if verbose:
            print 'Testing site %s' % names[i]
            print 'Amount remmoved %d' % amount
        hist = histograms[i]

        # This is used for cluster recommendations
        #elem, recommendedColor = recommender.recommendFromCluster(hist, xTrain)
            #if verbose:
        #   print 'Recommended from website %s' % names[elem]

        # This is used for vanilla classifiers
        cluster_names = recommender.clusterNames(hist)
        clusters.append(cluster_names)
        recommendedColor = recommender.predict(hist)
        r1, g1, b1 = image.binToRGB(color)
        r2, g2, b2 = image.binToRGB(recommendedColor)
        if verbose:
            print 'Removed color %d %d %d. Recommended color %d %d %d.' % (r1, g1, b1, r2, g2, b2)
            print 'Color distance: %d' % (image.binDistance(recommendedColor, color))
        recommendedColors[i] = recommendedColor

        # for plotting purposes
        if plot:
            colorRemoved.append(color)
            colorRecommend.append(recommendedColor)
            namesRecommend.append(names[i])
            #clusterIndex = recommender.returnClusterTest(hist)
            #clusterNames = recommender.clusterNames[clusterIndex]
            #clusterIndexList.append(clusterLin)
            #clusterLoc.append(clusterNames)

        if verbose:
            print 'Recommended color %d' % (recommendedColor)

        if recommendedColor == color:
            numCorrect += 1
    assert(len(clusters) == n)


    print 'Ignored: %d' % ignored
    print colorError(colors, recommendedColors)
    if plot:
        plotRecommend(colorRemoved, colorRecommend, namesRecommend, clusters)
    percentCorrect = float(numCorrect)/(n - ignored)
    return percentCorrect

def plotCluster(cluster, x, y, plotter):
    imagePath = data.getDataDir(amount=config.amount, cut=True, big=False)
    for i in xrange(len(cluster)):
        imager = mpimg.imread(imagePath + cluster[i])
        plotter.figimage(imager, i * 50 + x, y)

def plotRecommend(removed, recommend, names, clusters, xFactor=10, yFactor=10, myDpi=96):
    """
    removed: the color that we removed from the image
    recommend: the color that was recommended from the modified image
    names: the name of the image we were making a recommendation about
    clusterNames: the names of the images in the cluster we assigned this image to
    """
    imagePath = data.getDataDir(config.amount, cut=True, big=False)
    fig = plt.figure(figsize=(1024/myDpi, 1024/myDpi), dpi=myDpi)
    ax = fig.add_subplot(111)
    plt.axis("off")
    ctr = 0
    for i in xrange(len(names)):
        if ctr > SAMPLE_SIZE:
            break
        rr, rg, rb = image.binToRGB(removed[i])
        cr, cg, cb = image.binToRGB(recommend[i])
        rem = '#%02x%02x%02x' % (rr, rg, rb)
        rec = '#%02x%02x%02x' % (cr, cg, cb)
        #print rem
        #print rec
        SQ_SIZE = 30
        SQ_OFFSET = 55
        ax.add_patch(patches.Rectangle((125, i * SQ_OFFSET + 270),SQ_SIZE,SQ_SIZE, facecolor=rem))
        ax.add_patch(patches.Rectangle((175, i * SQ_OFFSET + 270),SQ_SIZE,SQ_SIZE, facecolor=rec))
        try:
            imager = mpimg.imread(imagePath + names[i])
            fig.figimage(imager, 125, i * 50 + 350)

            plotCluster(clusters[i], 500, i * 50 + 350, fig)
        except IOError:
            print "%s not found" % imagePath+names[i]
            pass
        ctr += 1
    ax.set_ylim([0,800])
    ax.set_xlim([0,800])
    plt.show()
        #remNorm = (float(rr)/256, float(rg)/256, float(rb)/256)
        #recNorm = (float(cr)/256, float(cg)/256, float(cb)/256)


def colorError(removed, recommended):
    n = len(removed)
    assert(len(recommended) == n)
    s = 0
    for i in xrange(n):
    	s += image.binSquareDistance(removed[i], recommended[i])
    return s / (n * 256 * 3)

def pickRandomColor(histogram):
    maxVal = np.amax(histogram)
    top_threshold = maxVal / 10.
    bottom_threshold = maxVal / 20.

    p = maxVal
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


r = ClusterRecommender(KMeans(n_clusters=200))
#r = ClusterRecommender(AffinityPropagation(damping=0.5))
print tester(histograms, r, verbose=True, plot=True)

#print 'Naive Bayes Classifier'
#print tester(histograms, GaussianNB(), verbose=False, plot=True)
