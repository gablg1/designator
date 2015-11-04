import numpy as np
from sklearn.cluster import KMeans
import math
import csv

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from ml_util import poly_features
from ml_util import simple_plot

def readCSV(filename):
    with open(filename, 'r') as csvfile:
        data = []
        names = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            names.append(row[0])
            data.append(row[1:])
        assert(len(names) == len(data))
        return names, data

colorgram = '../data/colorgram.csv'
website_names, X = readCSV(colorgram)
X = np.array(X)

N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

kmeans = KMeans(n_clusters = 8)
clusters = kmeans.fit_predict(X)
assert(len(clusters) == N)
websites = []
for i in range(len(clusters)):
    websites.append((clusters[i], website_names[i]))
websites.sort()
print websites

