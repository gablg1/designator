from PIL import Image
import csv

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
import numpy as np
import data
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

# we only grab the first 768 elements of each histogram, since
# we don't care about opacity
D = 768
histograms = [Image.open(path+img).histogram()[:D] for img in imgs]
assert(len(histograms) == len(imgs))
print "Each histogram has %d elements" % len(histograms[0])
print 'Normalizing them...'

assert(np.sum(ml.normalizeData(np.array(histograms[0]))) == 1.)
norm_histograms = [ml.normalizeData(np.array(h)) for h in histograms]

to = path + 'band-histograms.csv'
with open(to, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(len(norm_histograms)):
    	j = imgs[i].find('.')
        rank, name = int(imgs[i][:j]), imgs[i][j+1:]
        writer.writerow([rank, name] + norm_histograms[i].tolist())
