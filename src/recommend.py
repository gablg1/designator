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
from data import image

# We're using color histograms to represent websites
# x is 1 x D and cluster is N x D
# D = (256/bin_size)^3
def recommendFromCluster(x, cluster):
    N, D = cluster.shape
    assert(x.shape = (D, ))
    m = 0
    min_diff = 100000
    for i in range(N):
        diff = np.linalg.norm(x - cluster[i], 2)
        if diff < min_diff:
            min_diff = diff
            m = i
    return recommendFromElement(x, cluster[m])

# Takes in two 1 x D image vectors and recommends
# a color and how much should be added to the first one
def recommendFromElement(x, y):
    diff = y - x
    am = np.argmax(diff)
    return am, diff[am]



