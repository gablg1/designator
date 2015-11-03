from PIL import Image
import os
import csv

# default to current directory
path = "screenshots/"

fileList = os.listdir(path)
fileExt = ".png"
# it is currently expected that the files to be histogrammed lie in the
# same directory as histogrammer.py
imgs = filter(lambda File: File[-4:] == fileExt, fileList)

result = [Image.open(path+img).histogram() for img in imgs]

with open('colorgram.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in xrange(len(result)):
        writer.writerow([imgs[i]] + result[i])
