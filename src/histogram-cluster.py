import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import math
import csv
import matplotlib.image as mpimg
from sklearn.externals import joblib

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from ml_util import poly_features
from ml_util import simple_plot
from data import data
import config

amount=config.amount

def cluster(model):
    ranks, website_names, X = data.getBinnedHistograms(amount=amount, cut=config.cut, big=config.big)

    N, D = X.shape
    print "Each feature vector has dimension %d" % D
    print "Training on %d samples" % N

    clusters = model.fit_predict(X)
    assert(len(clusters) == N)
    websites = []
    for i in range(len(clusters)):
        websites.append((clusters[i], website_names[i]))
    websites.sort()
    print websites
    numClusters = len(set(clusters))
    print 'Number of clusters is %d' % numClusters

    data.plotClusters(websites, amount, clusters=numClusters, xFactor=75, yFactor=25)

# Writes kmeans object to pickle
    to = amount + '-histogram-clusters'
    pickle_to = '../persist/%s.pkl' % to
    joblib.dump(model, pickle_to)

# Writes clusters to csv
    with open(to + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(len(clusters)):
            writer.writerow([clusters[i], ranks[i], website_names[i]])

cluster(AffinityPropagation(damping=0.8))
#cluster(KMeans(n_clusters=50))

