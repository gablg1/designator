import numpy as np
#from sklearn.externals import joblib
#from sklearn.cluster import KMeans

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import data
#from data import image
import config
from ml_util import ml
from recommender import Recommender
from collections import defaultdict

class ClusterRecommender(Recommender):
    def __init__(self, model, amount=config.amount, cluster_type=config.cluster_type, highFactor=.1):
        self.model = model
        self.amount = amount
        self.cluster_type = cluster_type
        self.highFactor = highFactor
        self.ranks, self.names, self.histograms = data.getBandHistograms(amount, cut=True, big=False)

    # TODO: find another way to keep track of clusters and the name of the images that
    # belong in those cluster, pretty bad that we need to pass names to fit
    def fitWithPlot(self, train_data, target_classes, names):
        self.train_data = train_data

        result = self.model.fit_predict(train_data)

        self.cluster_names = ml.clusterResultsToNonNpArray(names, result)
        self.clusters = ml.clusterResultsToArray(train_data, result)


    def fit(self, train_data, target_classes):
        self.train_data = train_data
        result = self.model.fit_predict(train_data)
        self.clusters = ml.clusterResultsToArray(train_data, result)
        self.swan = self.findSwanAttribute(self.clusters[0])

    def clusterNames(self, x):
        p = self.model.predict(x)
        print self.cluster_names[p]
        return self.cluster_names[p]

    def cluster(self, x):
        p = self.model.predict(x)
        return self.clusters[p]

    def predict(self, x):

        C = self.cluster(x)
        return self.uglyDucklingRecommend(x, C)

    # Takes in two 1 x D image vectors and recommends
    # a color
    def recommendFromElement(self, x, y):
        diff = y - x
        am = np.argmax(diff)
        return am

    # We're using color histograms to represent websites
    # x is 1 x D and cluster is N x D
    # D = (256/bin_size)^3
    # The last argument is just to make tester.py work (poorly written code)
    def naiveRecommendFromCluster(self, x, cluster):
        N, D = cluster.shape
        assert(x.shape == (D,))
        m = 0
        min_diff = 100000
        for i in range(N):
            diff = ml.euclideanDistance(x, cluster[i])
            if diff < min_diff:
                min_diff = diff
                m = i
        a = self.recommendFromElement(x, cluster[m])
        return a

    def findSwanAttribute(self, cluster):
        _, D = cluster.shape
        means = np.array([0 for i in xrange(D)])
        for d in xrange(D):
            samples = []
            for elem in cluster:
                samples.append(elem[d])
            means[d] = np.mean(np.array(samples))
        return np.argmax(means)

    def uglyDucklingRecommend(self, x, cluster):
        N, D = cluster.shape
        assert(x.shape == (D,))

        means = np.mean(cluster, axis=0)
        varss = np.var(cluster, axis=0)
        for d in xrange(D):
            if x[d] > 0:
            	means[d] = 0
            if means[d] == 0 or varss[d] == 0:
            	varss[d] = 1
        return np.argmax(means)

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
