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
import tester
from ml_util import ml
from cluster_recommender import ClusterRecommender
from random_recommender import RandomRecommender
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

amount = config.amount
ranks, names, histograms = data.getBinnedHistograms(amount, cut=True, big=False)
SAMPLE_SIZE = 10

def show(data, recommender, fractionTrain=.8, highFactor=.1, verbose=False, plot=False):
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

    train_colors, _, train_histograms = tester.removeColors(xTrain, highFactor=highFactor)
    try:
        recommender.fitWithPlot(train_histograms, train_colors, train_names)
    except:
        print train_colors
        recommender.fit(train_histograms, train_colors)
    if verbose:
    	print 'Done fitting'

    colors, quantities, histograms = tester.removeColors(xTest, highFactor=highFactor)
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
        try:
            cluster_names = recommender.clusterNames(hist)
            clusters.append(cluster_names)
        except:
            pass

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


    print 'Ignored: %d' % ignored
    print tester.colorError(colors, recommendedColors)
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
        SQ_SIZE = 50
        SQ_OFFSET = 82
        BASE_OFFSET = 80
        ax.add_patch(patches.Rectangle((125, i * SQ_OFFSET),SQ_SIZE,SQ_SIZE, facecolor=rem))
        ax.add_patch(patches.Rectangle((175, i * SQ_OFFSET),SQ_SIZE,SQ_SIZE, facecolor=rec))
        try:
            imager = mpimg.imread(imagePath + names[i])
            fig.figimage(imager, 125, i * 50 + BASE_OFFSET)

            if len(clusters) > 0:
                plotCluster(clusters[i], 300, i * 50 + BASE_OFFSET, fig)
        except IOError:
            print "%s not found" % imagePath+names[i]
            pass
        ctr += 1
    ax.set_ylim([0,1024])
    ax.set_xlim([0,1024])
    plt.show()
        #remNorm = (float(rr)/256, float(rg)/256, float(rb)/256)
        #recNorm = (float(cr)/256, float(cg)/256, float(cb)/256)



#r = ClusterRecommender(KMeans(n_clusters=50))
r = ClusterRecommender(AffinityPropagation(damping=0.8))

#r = ClusterRecommender(KMeans(n_clusters=1))
print show(histograms, r, verbose=True, plot=True)

#print show(histograms, RandomForestClassifier(), verbose=True, plot=True)


