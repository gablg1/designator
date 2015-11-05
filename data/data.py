import os
import csv
import numpy as np

def readCSV(filename):
    with open(filename, 'r') as csvfile:
        data = []
        names = []
        ranks = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ranks.append(row[0])
            names.append(row[1])
            data.append(row[2:])
        assert(len(names) == len(data))
        return ranks, names, np.array(data)

def getHistogram(amount, cut, big):
    dirpath = getDataDir(amount, cut, big)
    filepath = '%s/%s' % (dirpath, 'colorgram.csv')
    return readCSV(filepath)

def getDataDir(amount, cut, big):
    if cut and big:
        path = "cut_screenshots"
    elif cut and not big:
        path = "small_cut_screenshots"
    elif not cut and big:
        path = 'screenshots'
    elif not cut and not big:
        path = 'small_screenshots'

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return '%s/%s/%s/' % (cur_dir, amount, path)
