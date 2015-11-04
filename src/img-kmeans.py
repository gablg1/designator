from PIL import Image
from sklearn.cluster import KMeans
import os
import numpy as np

toSize = 68,38
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

path = '../data/' + path

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

kmeans = KMeans(n_clusters = 8)
clusters = kmeans.fit_predict(X)
assert(len(clusters) == N)
websites = []
for i in range(len(clusters)):
    websites.append((clusters[i], website_names[i]))
websites.sort()
print websites

