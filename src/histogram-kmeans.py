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

def readCSV(filename):
    with open(filename, 'r') as csvfile:
        data = []
        names = []
        ranks = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ranks.append(row[0])
            names.append(row[1])
            data.append(row[2:])
        assert(len(names) == len(data))
        return ranks, names, data

amount = 'top-15k'
CUT = True
BIG = True
# choose the screenshots directory
if CUT and BIG:
    path = "cut_screenshots/"
elif CUT and not BIG:
    path = "small_cut_screenshots/"
elif not CUT and BIG:
    path = 'screenshots/'
elif not CUT and not BIG:
    path = 'small_screenshots/'

path = '../data/%s/%s' % (amount, path)
colorgram = path + 'colorgram.csv'

ranks, website_names, X = readCSV(colorgram)
X = np.array(X)

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


imagePath = "../data/small_cut_screenshots/"
def plotClusters(xoffset=10, yoffset=10, my_dpi=96):
    """
     We want to plot every image according to the appropriate point
     on the x-axis according to its cluster number. We want to plot
     each new member of a given cluster at a higher y position
    """
    clusterDict = [0 for n in xrange(numClusters)]
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    for site in websites:
        clusterIndex, address = site
        try:
            image = mpimg.imread(imagePath+address)
            y = clusterDict[clusterIndex] * yoffset
            clusterDict[clusterIndex]+=1
            plt.figimage(image, clusterIndex*xoffset, y)
        except IOError:
            # usually if we don't have the small cut image yet
            pass

    plt.show()

#plotClusters(xoffset=75, yoffset=25)



