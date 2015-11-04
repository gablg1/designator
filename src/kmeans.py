import numpy as np
from sklearn.cluster import KMeans
import math

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from ml_util import poly_features
from ml_util import simple_plot

data = np.genfromtxt('../data/colorgram.csv', delimiter=',')

websites = data[:, 0]
X = data[:, 1:]

N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

kmeans = KMeans(n_clusters = 8)
print kmeans.fit_predict(X)
