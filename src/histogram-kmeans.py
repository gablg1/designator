import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import csv
import matplotlib.image as mpimg

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from ml_util import poly_features
from ml_util import simple_plot
from data import data


amount='top-15k'
ranks, website_names, X = data.getHistogram(amount=amount, cut=True, big=True)

N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

numClusters = 8
kmeans = KMeans(n_clusters = numClusters)
clusters = kmeans.fit_predict(X)
assert(len(clusters) == N)
websites = []
for i in range(len(clusters)):
    websites.append((clusters[i], website_names[i]))
websites.sort()
print websites


data.plotClusters(websites, xFactor=75, yFactor=25)
