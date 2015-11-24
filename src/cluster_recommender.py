class Recommender():
    # subclass has to override this
    def train(self, train_data, target_classes):
        self.train_data = train_data
        self.train_targets = target_classes

    # subclass has to override this
    # returns a color
    def recommend(self, x):
        return 0
