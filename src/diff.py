from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from data import data

# gets Screenshots directory as string
BIG = True
amount='top-15k'
path = data.getDataDir(amount=amount, cut=True, big=BIG)

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

x_file = '7993.gandi.net.png'
if BIG:
	x_file = 'big_' + x_file
else:
	x_file = 'small_' + x_file
x = imgToArray(x_file)

start = time.time()
m = 0
min_diff = 100000
for i in range(N):
    diff = np.linalg.norm(x - X[i], 2)
    if diff < min_diff:
        min_diff = diff
        m = i
print m, min_diff, imgs[m]
print x - X[m]
print np.linalg.norm(x - X[m], 2)
end = time.time()
print 'Time elapsed: %f' % (end - start)

