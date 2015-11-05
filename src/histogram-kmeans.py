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


dataPath = data.getHistogram(amount, True, True)
data.plotClusters(dataPath, websites, xFactor=75, yFactor=25, sampleSize=3)



#imagePath = "../data/60/small_cut_screenshots/"
#def plotClusters(xFactor=10, yFactor=10, myDpi=96, sampleSize=100):
    #"""
     #We want to plot every image according to the appropriate point
     #on the x-axis according to its cluster number. We want to plot
     #each new member of a given cluster at a higher y position
    #"""
    #clusterDict = [0 for n in xrange(numClusters)]
    #plt.figure(figsize=(800/myDpi, 800/myDpi), dpi=myDpi)
    #for site in websites:
        #clusterIndex, address = site
        #try:
            #yIndex = clusterDict[clusterIndex]
            #if yIndex > sampleSize:
                #pass
            #image = mpimg.imread(imagePath+address)
            #y = yIndex * yFactor
            #clusterDict[clusterIndex]+=1
            #plt.figimage(image, clusterIndex*xFactor, y)
        #except IOError:
            ## usually if we don't have the small cut image yet
            #pass

    #plt.show()

#plotClusters(xFactor=75, yFactor=25, sampleSize=3)



