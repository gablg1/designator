import numpy as np
from sklearn.cluster import KMeans
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
ranks, website_names, X = data.getBandHistograms(amount=amount, cut=config.cut, big=config.big)

N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

def euclideanDistance(a, b):
    return np.linalg.norm(a - b)


# Similarities matrix
S = np.zeros((N, N))
for i in xrange(N):
    for j in xrange(N):
        S[i, j] = -euclideanDistance(X[i], X[j])

# Preference i p(i) = S(i,i)
# This is the a priori suitability of point i to serve as an exemplar
# We initialize all of them to the median
median = np.median(S)
for i in xrange(N):
	S[i, i] = median
print S
# Responsabilities matrix
R = np.zeros((N, N))

# Availability matrix
A = np.zeros((N, N))

C = np.array(xrange(N))

for epoch in xrange(10):
    print epoch
    # Responsibility updates
    print 'R before'
    print R
    for i in xrange(N):
        for k in xrange(N):
            max_sum = 0
            for j in xrange(N):
                val = A[i, j] + S[j, i]
                if j != k and val > max_sum:
                    max_sum = val
            R[i, k] = S[i, k] - max_sum
    print 'R after'
    print R

    print 'A before'
    print A
    # Availability updates
    for i in xrange(N):
        for k in xrange(N):
            term = 0
            for j in xrange(N):
                if j != i and j != k:
                    term += max(0, R[j, k])

            if i == k:
                A[k, k] = term
            else:
                A[i, k] = min(0, R[k, k] + term)
    print 'A after'
    print A

    # Cluster assignments
    #RAp = R + A.T
    for i in xrange(N):
        argmax = 0
        valmax = 0
        for k in xrange(N):
            val = R[i, k] + A[k, i]
            if val > valmax:
                valmax = val
                argmax = k

        #c = np.argmax(RAp[i, :])
        C[i] = argmax
        print i, C[i]

print S
