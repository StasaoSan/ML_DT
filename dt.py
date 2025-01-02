import numpy as np
from start_tests import run_experiment_custom

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DTclassifier_custom:
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _entropy(self, y):
        m = len(y)
        entropy = 0.0
        for c in np.unique(y):
            p = np.sum(y == c) / m
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        if self.criterion == 'gini':
            parent_score = self._gini(y)
        elif self.criterion == 'entropy':
            parent_score = self._entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]

                if self.criterion == 'gini':
                    score_left = self._gini(y_left)
                    score_right = self._gini(y_right)
                else:
                    score_left = self._entropy(y_left)
                    score_right = self._entropy(y_right)

                n_left, n_right = len(y_left), len(y_right)
                gain = parent_score - (n_left / m) * score_left - (n_right / m) * score_right

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionNode(value=predicted_class)

        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) == 1 or \
           len(y) < self.min_samples_split:
            return node

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return node

        indices_left = X[:, feature_index] <= threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        node.value = None

        return node

    def _predict(self, inputs):
        node = self.root
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def get_tree_depth(self):
        return self._get_depth(self.root)

    def _get_depth(self, node):
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def evaluate_with_min_samples_leaf(self, X_train, y_train, X_val, y_val):
        hyperparam_name = "min_samples_leaf"
        hyperparam_values = range(1, 21)
        fixed_params = {
            "max_depth": None,
            "criterion": "gini",
            "min_samples_split": 2
        }
        run_experiment_custom(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params)

    def evaluate_min_samples_split(self, X_train, y_train, X_val, y_val):
        hyperparam_name = "min_samples_split"
        hyperparam_values = range(2, 21)
        fixed_params = {
            "max_depth": None,
            "criterion": "gini",
            "min_samples_leaf": 1
        }
        run_experiment_custom(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params)

    def evaluate_with_criterion(self, X_train, y_train, X_val, y_val):
        hyperparam_name = "criterion"
        hyperparam_values = ["gini", "entropy"]
        fixed_params = {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        run_experiment_custom(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params)

    def evaluate_with_max_depth(self, X_train, y_train, X_val, y_val):
        hyperparam_name = "max_depth"
        hyperparam_values = list(range(1, 21)) + [None]
        fixed_params = {
            "criterion": "gini",
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        run_experiment_custom(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params, plot_type="accuracy")
