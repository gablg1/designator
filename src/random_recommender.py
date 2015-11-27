import numpy as np
import random
from sklearn.externals import joblib
from sklearn.cluster import KMeans

# Hack to import ml_util from the parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import data
from data import image
import config
from ml_util import ml
from recommender import Recommender

class RandomRecommender(Recommender):
    def __init__(self, amount=config.amount, cluster_type=config.cluster_type):
        self.amount = amount
        self.cluster_type = cluster_type

    # subclass has to override this
    def fit(self, train_data, target_classes):
        self.train_data = train_data


    def predict(self, x):
        D = x.shape[0]
        zeros = []
        for d in xrange(D):
            if x[d] <= 0:
            	zeros.append(d)
        return random.choice(zeros)

