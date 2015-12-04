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

path = data.getDataDir(config.amount, config.cut, config.big)

# gets all files
fileList = os.listdir(path)
fileExt = ".png"
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)


ranks, names, histograms = data.getBinnedHistograms(amount, cut=True, big=False)

X = np.array([image.imgToArray(path+imgs[i]) for i in xrange(len(imgs))])
print X.shape, histograms.shape
website_names = imgs

BIN_HISTOGRAMS = {}
IMG_ARRAYS = {}
assert(len(names) == len(website_names))
for i in xrange(len(names)):
    assert(website_names[i] == names[i])
    IMG_ARRAYS[names[i]] = X[i]
    BIN_HISTOGRAMS[names[i]] = histograms[i]
names = np.array(names)

def test(data, names, recommender, fractionTrain=.8, highFactor=.1, verbose=False):
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
    namesTrain, namesTest = ml.splitData(names, fractionTrain)
    n = xTest.shape[0]
    m = xTrain.shape[0]

    train_colors, _, train_data, train_histograms = removeColors(xTrain, namesTrain)
    print 'Done removing train colors'
    try:
        recommender.fit(train_data, train_colors, train_histograms)
    except:
        recommender.fit(train_data, train_colors)
    print 'Done fitting'

    colors, quantities, test_imgs, test_histograms = removeColors(xTest, namesTest)
    n = colors.shape[0]
    assert(test_imgs.shape[0] == n)
    print 'Done removing test colors'

    numCorrect = 0

    D = BIN_HISTOGRAMS[names[0]].shape[0]
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
            print 'Partial %d: %f' % (i, float(numCorrect) / i)

        color, amount = colors[i], quantities[i]
        # Ignore colors that might bias us
        if count[color] > color_mean + color_stdev:
            ignored += 1
            continue

        img, hist = test_imgs[i], test_histograms[i]
        # Ignore colors that are basically the background
        if hist[color] > 0.4:
        	ignored += 1
        	continue

        try:
            cluster = recommender.cluster(img)
            intersectionRatio += core.clusterIntersectionRatio(hist, cluster)
        except:
            pass

        if verbose:
            print 'Testing site %s' % namesTest[i]
            print 'Amount remmoved %f' % amount

        try:
            recommendedColor = recommender.predictImg(img, hist)
        except:
            recommendedColor = recommender.predict(img)
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

def replaceColor(imgArray, fro, to):
    r, g, b = image.binToRGB(fro)
    r2, g2, b2 = image.binToRGB(to)
    if r2 == 250 and g2 == 250 and b2 == 250:
        r2, g2, b2 = 255, 255, 255
    i = 0
    pixels_removed = 0
    while i + 2 < len(imgArray):
        ri, gi, bi = imgArray[i], imgArray[i+1], imgArray[i+2]
        b = image.RGBToBin(ri, gi, bi)
        if b == fro:
            imgArray[i], imgArray[i+1], imgArray[i+2] = r2, g2, b2
            pixels_removed += 1
        i += 3
    return imgArray, pixels_removed


def removeColors(data, names):
    N = data.shape[0]
    assert(N == len(names))
    ret = []
    binned_ret = []
    colorsRemoved = []
    quantityRemoved = []
    for i in xrange(N):
        binned_histogram = BIN_HISTOGRAMS[names[i]]
        color = pickRandomColor(binned_histogram)

        # actually removes the color
        most_common = np.argmax(binned_histogram)
        old = np.copy(data[i])
        new_img, px_removed = replaceColor(data[i], color, most_common)
        # This assert is a little too harsh
        #assert(px_removed > 0)
        if px_removed == 0:
        	continue

        colorsRemoved.append(color)
        quantityRemoved.append(binned_histogram[color])
        ret.append(new_img)
        binned_histogram[color] = 0
        binned_ret.append(binned_histogram)
    return np.array(colorsRemoved), np.array(quantityRemoved), np.array(ret), np.array(binned_ret)

if __name__ == '__main__':
    print 'Kmeans Classifier'
    r = ClusterRecommender(KMeans(n_clusters=15))
    print test(X, names, r, verbose=False)

    print 'Affinity Propagation Classifier'
    r = ClusterRecommender(AffinityPropagation(damping=0.8))
    print test(X, names, r, verbose=False)

    print 'Affinity Propagation Classifier'
    r = ClusterRecommender(AffinityPropagation(damping=0.99))
    print test(X, names, r, verbose=False)

    print 'Whole Set Classifier'
    r = ClusterRecommender(KMeans(n_clusters=1))
    print test(X, names, r, verbose=False)

    print 'Naive Bayes Classifier'
    print test(X, names, GaussianNB(), verbose=False)


    print 'Random Forest Classifier'
    print test(X, names, RandomForestClassifier())

#print 'Affinity Propagation Classifier'
#r = ClusterRecommender(AffinityPropagation(damping=0.7))
#print test(histograms, r, verbose=False)
