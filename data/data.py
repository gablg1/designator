import os
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def getRankFromFilename(filename):
    found = filename.find('.')
    return int(found[:found])

def readCSV(filename):
    with open(filename, 'r') as csvfile:
        data = []
        names = []
        ranks = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ranks.append(int(row[0]))
            names.append(row[0] + '.' + row[1])
            data.append(row[2:])
        assert(len(names) == len(data))
        return ranks, names, np.array(data).astype(float)

# these histograms are by band. Aka Dimension = 256 * 3
def getBandHistograms(amount, cut, big):
    dirpath = getDataDir(amount, cut, big)
    filepath = '%s%s' % (dirpath, 'band-histograms.csv')
    return readCSV(filepath)

def getBinnedHistograms(amount, cut, big):
    dirpath = getDataDir(amount, cut, big)
    filepath = '%s%s' % (dirpath, 'histograms.csv')
    return readCSV(filepath)

def getDataDir(amount, cut, big):
    if cut and big:
        path = "cut_screenshots"
    elif cut and not big:
        path = "small_cut_screenshots"
    elif not cut and big:
        path = 'screenshots'
    elif not cut and not big:
        path = 'small_screenshots'

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return '%s/%s/%s/' % (cur_dir, amount, path)

def plotClusters(websites, amount, clusters=8,xFactor=10, yFactor=10, myDpi=96, sampleSize=20):
    """
     We want to plot every image according to the appropriate point
     on the x-axis according to its cluster number. We want to plot
     each new member of a given cluster at a higher y position

     Inputs:
         imagePath is a string to where the images are located
         websites is a list of tuples of the form:
             (clusterNumber, websitename)
    """
    imagePath = getDataDir(amount, cut=True, big=False)
    clusterDict = [0 for n in xrange(clusters)]
    plt.figure(figsize=(800/myDpi, 800/myDpi), dpi=myDpi)
    for site in websites:
        clusterIndex, address = site
        try:
            yIndex = clusterDict[clusterIndex]
            if yIndex > sampleSize:
                continue
            image = mpimg.imread(imagePath+address)
            y = yIndex * yFactor
            clusterDict[clusterIndex]+=1
            plt.figimage(image, clusterIndex*xFactor, y)
        except IOError:
            # usually if we don't have the small cut image yet
            pass

    plt.show()

def readClustersAsDict(filename):
    with open(filename, 'r') as csvfile:
        current = 0
        clusters = {}
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            cluster = int(row[0])
            rank = int(row[1])
            clusters[rank] = cluster
        return clusters



def readClusters(filename):
    with open(filename, 'r') as csvfile:
        current = 0
        clusters = [[]]
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            cluster = int(row[0])
            if current < cluster:
                assert(current == cluster - 1)
                current += 1
                clusters.append([])
            assert(current == cluster)
            clusters[current].append((int(row[1]), row[2]))
        return clusters
