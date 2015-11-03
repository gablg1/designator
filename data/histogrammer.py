from PIL import Image
import csv

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
import numpy as np

# default to current directory
path = "screenshots/"

fileList = os.listdir(path)
fileExt = ".png"
# it is currently expected that the files to be histogrammed lie in the
# same directory as histogrammer.py
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

histograms = [Image.open(path+img).histogram() for img in imgs]
assert(len(histograms) == len(imgs))

norm_histograms = [ml.normalizeData(np.array(h)) for h in histograms]

with open('colorgram.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(len(norm_histograms)):
        writer.writerow([imgs[i]] + norm_histograms[i].tolist())
