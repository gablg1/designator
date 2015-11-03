from PIL import Image
import os

toSize = 68,38
# default to screenshots directory
path = "screenshots/"

fileList = os.listdir(path)
fileExt = ".png"
# it is currently expected that the files to be histogrammed lie in the
# same directory as histogrammer.py
imgs = filter(lambda File: File[-4:] == fileExt, fileList)
imgs.sort()
print "Found %d %s images" % (len(imgs), fileExt)

for i in xrange(len(imgs)):
    splitName = imgs[i].rpartition('.')
    assert(splitName[0] != "")
    assert(splitName[1] != "")
    newName = "small_" + splitName[0] + splitName[1] + splitName[2]
    try:
        Image.open(path+imgs[i]).resize(toSize).save(path + newName)
    except IOError as e:
        print e
