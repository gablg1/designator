import numpy as np
import math

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ml_util import ml
from ml_util import poly_features
from ml_util import simple_plot

data = np.genfromtxt('../data/colorgram.csv', delimiter=',')

websites = data[:, 0]
X = data[:, 1:]

print X[5]
N, D = X.shape
print "Each feature vector has dimension %d" % D
print "Training on %d samples" % N

# Right now we just do a simple bar plot of the histogram
plt = simple_plot.Plotter()
plt.plotBar(X[5])
plt.show()
