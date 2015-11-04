from PIL import Image
import os

toSize = 68,38
CUT = True
# default to screenshots directory
if CUT:
    path = "cut_screenshots/"
    to_path = 'small_cut_screenshots/'
else:
    path = "screenshots/"
    to_path = 'small_screenshots/'

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
