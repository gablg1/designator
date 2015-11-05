import os
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
        return ranks, names, np.array(data)

def getHistogram(amount, cut, big):
    dirpath = getDataDir(amount, cut, big)
    filepath = '%s/%s' % (dirpath, 'colorgram.csv')
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

def plotClusters(imagePath, websites, numClusters=8,xFactor=10, yFactor=10, myDpi=96, sampleSize=100):
    """
     We want to plot every image according to the appropriate point
     on the x-axis according to its cluster number. We want to plot
     each new member of a given cluster at a higher y position

     Inputs:
         imagePath is a string to where the images are located
         websites is a list of tuples of the form:
             (clusterNumber, websitename)
    """
    clusterDict = [0 for n in xrange(numClusters)]
    plt.figure(figsize=(800/myDpi, 800/myDpi), dpi=myDpi)
    for site in websites:
        clusterIndex, address = site
        try:
            yIndex = clusterDict[clusterIndex]
            print "yindex is : " + str(yIndex)
            print "sampleSize is: " + str(sampleSize)
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


