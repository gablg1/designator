from PIL import Image
import csv

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
import numpy as np
import data
import image
import config

# choose the screenshots directory
amount = config.amount
path = data.getDataDir(amount=amount, cut=config.cut, big=config.big)

fileList = os.listdir(path)
fileExt = ".png"
# it is currently expected that the files to be histogrammed lie in the
# same directory as histogrammer.py
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

histograms = []
for i in xrange(len(imgs)):
    hist = image.imgToBinnedHistogram(path + imgs[i])
    if np.count_nonzero(hist) < 3:
    	print hist
        print 'Removing', imgs[i]
        os.remove(path + imgs[i])
    else:
        histograms.append(ml.normalizeData(np.array(hist)))
print "Each histogram has %d elements" % len(histograms[0])

to = path + 'histograms.csv'
with open(to, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(len(histograms)):
        j = imgs[i].find('.')
        rank, name = int(imgs[i][:j]), imgs[i][j+1:]
        writer.writerow([rank, name] + histograms[i].tolist())
