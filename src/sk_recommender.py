from recommender import Recommender

class SKRecommender(Recommender):
    def __init__(self, model):
        self.model = model()

    # Trains naive bayes recommender
    def train(self, train_data, target_classes):
        self.model.fit(train_data, target_classes)

    # Should be called after train
    def recommend(self, histogram):
        assert(self.model)
        return self.model.predict(histogram)
