import numpy as np
from sklearn.externals import joblib


# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import data
from data import image
import config
from ml_util import ml

def recommend(img_path, amount=config.amount, cluster_type=config.cluster_type):
    # First we get the histogram of the data
    x = image.imgToBinnedHistogram(img_path)

    # Load both the kmeans object and the already calculated clusters
    kmeans = joblib.load('./../persist/%s-%s.pkl' % (amount, cluster_type))
    clusters = data.readClustersAsDict('%s-%s.csv' % (amount, cluster_type))
    print 'done'

    # Find where in clusters the image should fit
    band_hist = image.imgToBandHistogram(img_path)
    print 'done'
    p = kmeans.predict(band_hist)
    print 'done'

    ranks, names, histograms = data.getHistograms(amount, cut=True, big=False)

    # We recreate the clusters containing histograms
    C = []
    for i in xrange(len(histograms)):
        # sites are uniquely identified by rank
        c = clusters[ranks[i]]
        if c == p:
            C.append(histograms[i])
    C = np.array(C)
    print C

    print C.shape
    return recommendFromCluster(x, C)

