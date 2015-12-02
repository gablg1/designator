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
from recommender import Recommender
from collections import defaultdict

class DucklingRecommender(Recommender):
    def __init__(self, cluster_size=15, amount=config.amount):
        self.amount = amount
        self.cluster_size = cluster_size

    # TODO: find another way to keep track of clusters and the name of the images that
    # belong in those cluster, pretty bad that we need to pass names to fit
    def fitWithPlot(self, train_data, target_classes, names):
        self.train_data = train_data

        result = self.model.fit_predict(train_data)

        self.cluster_names = ml.clusterResultsToNonNpArray(names, result)
        self.clusters = ml.clusterResultsToArray(train_data, result)

    def fit(self, train_data, target_classes, histograms=[]):
        self.train_data = train_data

    def testClusters(self, x):
        p = self.model.predict(x)
        print 'Number of clusters: ', len(self.clusters)
        for i in xrange(len(self.clusters)):
        	print core.clusterIntersectionRatio(x, self.clusters[i])
        C = self.clusters[p]
        print 'This should be better:'
        print core.clusterIntersectionRatio(x, C)
        print 'This should be even better:'
        D = self.buildCluster(x)
        print core.clusterIntersectionRatio(x, D)


    def clusterNames(self, x):
        p = self.model.predict(x)
        return self.cluster_names[p]

    def cluster(self, x):
        return self.buildCluster(x, self.cluster_size)

    def predictImg(self, imgArray, hist):
        # We use the image for clustering
        C = self.cluster(imgArray)

        # and the corresponding histogram for ugly duckling
        return core.uglyDucklingRecommend(hist, C)

    def predict(self, x):
        C = self.buildCluster(x, self.cluster_size)
        return core.uglyDucklingRecommend(x, C, var=False)

    def buildCluster(self, x, size):
        n = len(self.train_data)
        ratio = np.zeros(n)
        for i in xrange(n):
       	    ratio[i] = core.intersectionRatio(x, self.train_data[i])
       	max_args = np.argpartition(ratio, -size)[-size:]

        return self.train_data[max_args]


    # given an image name, return what cluster the image is in
    # if this is something we've trained on, then train should
    # be true and we will just re-predict where it should be
    def returnClusterTrain(self, name):
        where = self.names.index(name)
        hist = self.histograms[where]
        posInCluster = self.model.predict(hist)
        return posInCluster[0]

    # given histogram data, return what cluster we assign this
    # image to
    def returnClusterTest(self, hist):
        posInCluster = self.model.predict(hist)
        return posInCluster[0]
