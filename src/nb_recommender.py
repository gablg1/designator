from sklearn.naive_bayes import GaussianNB

class NBRecommender():
    # Trains naive bayes recommender
    def train(self, train_data, target_classes):
        self.gnb = GaussianNB()
        self.gbn.fit(train_data, target_classes)

    # Should be called after train
    def recommend(x):
        assert(self.gnb)
        return self.gnb.predict(x)
