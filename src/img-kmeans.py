from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from sklearn.externals import joblib

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from data import data
from data import image

import config

# gets Screenshots directory as string
amount = config.amount
path = data.getDataDir(config.amount, config.cut, config.big)

# gets all files
fileList = os.listdir(path)
fileExt = ".png"
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

X = np.array([image.imgToArray(path+imgs[i]) for i in xrange(len(imgs))])
print X
website_names = imgs

N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

numClusters = 20
kmeans = KMeans(n_clusters = numClusters)
clusters = kmeans.fit_predict(X)
assert(len(clusters) == N)
websites = []
for i in range(len(clusters)):
    websites.append((clusters[i], website_names[i]))
websites.sort()
print websites

data.plotClusters(websites, amount, clusters=numClusters, xFactor=75, yFactor=25)
to = '../persist/' + amount + '-img-kmeans.pkl'
joblib.dump(kmeans, to)
