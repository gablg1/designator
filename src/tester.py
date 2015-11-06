import numpy as np
# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from copy import deepcopy
import recommender
"""
expected recommender function signature is going to be recommendFromCluster(x, cluster): color, howMuch
the test plan is currently:
    given an cluster, separate into test and train sets
    take an element from the test set, subtract out the mean color
    our elements are going to be histograms of colors
    send this modified image to the recommendFromCluster function.
    take the recommendation and reapply it to the image, diff the newImage with the initial one and
    the rmse will be the sum of the difference squared

    the cluster will be a list of histograms
"""

highFactor = .2
lowFactor = .1


def tester(cluster, fractionTrain=.9):
    endTrain = round(cluster.shape[0] * fractionTrain)
    xTrain = cluster[0:endTrain,]
    xTest = cluster[endTrain:,]
    mse = 0.
    for testHist in xTest:
        testSum = np.sum(testHist)
        copiedHist = deepcopy(testHist)
        for i in xrange(len(testHist)):
            if testHist[i]/testSum < .1:
                continue
            elif testHist[i]/testSum >= .2:
                testHist[i]-= testSum * .2
                break
        color, howMuch = recommender.recommendFromCluster(testHist, xTrain)
        testHist[color]+= howMuch
        diff = copiedHist - testHist
        mse += np.sum(diff**2)
    rmse = np.sqrt(mse/len(xTest))
    return rmse

