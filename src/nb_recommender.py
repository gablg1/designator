from sklearn.naive_bayes import GaussianNB
from recommender import Recommender

class NBRecommender(Recommender):
    def __init__(self):
        self.gnb = GaussianNB()

    # Trains naive bayes recommender
    def train(self, train_data, target_classes):
        print train_data.shape, target_classes.shape
        self.gnb.fit(train_data, target_classes)

    # Should be called after train
    def recommend(self, histogram):
        assert(self.gnb)
        return self.gnb.predict(histogram)
