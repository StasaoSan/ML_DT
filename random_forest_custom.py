import numpy as np
from dt import DTclassifier_custom
from collections import Counter


class RandomForestClassifierCustom:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.features_indices = []

    def fit(self, X_train, y_train):
        np.random.seed(self.random_state)
        self.trees = []
        self.features_indices = []

        n_samples, n_features = X_train.shape

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
            else:
                X_sample = X_train
                y_sample = y_train

            features_idx = np.random.choice(n_features, max_features, replace=False)
            self.features_indices.append(features_idx)

            tree = DTclassifier_custom(max_depth=self.max_depth)

            tree.fit(X_sample[:, features_idx], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree, features_idx in zip(self.trees, self.features_indices):
            pred = tree.predict(X[:, features_idx])
            predictions.append(pred)

        predictions = np.array(predictions).T

        y_pred = []
        for preds in predictions:
            y_pred.append(Counter(preds).most_common(1)[0][0])

        return np.array(y_pred)
