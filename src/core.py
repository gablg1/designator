import numpy as np
#from sklearn.externals import joblib
#from sklearn.cluster import KMeans

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import data
from data import image
#from data import image
import config
import core
from ml_util import ml

def intersectionRatio(hist1, hist2):
    assert(ml.isVector(hist1) and ml.isVector(hist2))
    nz_1 = np.nonzero(hist1)[0]
    nz_2 = np.nonzero(hist2)[0]
    intersecting = len(np.intersect1d(nz_1, nz_2))
    dividend = len(nz_1) + len(nz_2) - intersecting
    if dividend == 0:
    	print len(nz_1), len(nz_2), intersecting
    	print hist1
    	print hist2
    ratio = float(intersecting) /(len(nz_1) + len(nz_2) - intersecting)
    return ratio

def clusterIntersectionRatio(hist, cluster):
    N, D = cluster.shape
    assert(hist.shape == (D,))
    ratio_sum = 0
    for i in xrange(N):
        ratio_sum += intersectionRatio(hist, cluster[i])
    return ratio_sum / N

# Takes in two 1 x D image vectors and recommends
# a color
def recommendFromElement(x, y):
    diff = y - x
    am = np.argmax(diff)
    return am

# We're using color histograms to represent websites
# x is 1 x D and cluster is N x D
# D = (256/bin_size)^3
# The last argument is just to make tester.py work (poorly written code)
def naiveRecommendFromCluster(x, cluster):
    N, D = cluster.shape
    assert(x.shape == (D,))
    m = 0
    min_diff = 100000
    for i in range(N):
        diff = ml.euclideanDistance(x, cluster[i])
        if diff < min_diff:
            min_diff = diff
            m = i
    a = recommendFromElement(x, cluster[m])
    return a

def uglyDucklingRecommend(x, cluster, var=False):
    N, D = cluster.shape
    assert(x.shape == (D,))

    means = np.mean(cluster, axis=0)
    varss = np.var(cluster, axis=0)
    for d in xrange(D):
        if x[d] > 0:
            means[d] = 0
        if means[d] == 0 or varss[d] == 0:
            varss[d] = 1
    if var:
        return np.argmax(means / varss)
    else:
        return np.argmax(means)
