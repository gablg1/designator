from PIL import Image
import os
import data
import config

toSize = 68,38

# choose the screenshots directory
amount = config.amount
cut = config.cut
path = data.getDataDir(amount, cut, big=True)
to_path = data.getDataDir(amount, cut, big=False)

fileList = os.listdir(path)
fileExt = ".png"
# it is currently expected that the files to be histogrammed lie in the
# same directory as histogrammer.py
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

for i in xrange(len(imgs)):
    try:
        Image.open(path+imgs[i]).resize(toSize).save(to_path + imgs[i])
    except IOError as e:
        print e
