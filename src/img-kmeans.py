from PIL import Image
from sklearn.cluster import KMeans
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data import data

# gets Screenshots directory as string
path = data.getDataDir(amount='top-15k', cut=True, big=False)

# gets all files
fileList = os.listdir(path)
fileExt = ".png"
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

def imgToArray(filepath):
    try:
        img = np.array(Image.open(path + imgs[i]))
    except IOError as e:
        print e
    return img[:, :, :3].ravel()

X = np.array([imgToArray(path+imgs[i]) for i in xrange(len(imgs))])
print X
website_names = imgs

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
            image = mpimg.imread(path+address)
            y = clusterDict[clusterIndex] * yoffset
            clusterDict[clusterIndex]+=1
            plt.figimage(image, clusterIndex*xoffset, y)
        except IOError:
            # usually if we don't have the small cut image yet
            pass

    plt.show()

plotClusters(xoffset=75, yoffset=25)


