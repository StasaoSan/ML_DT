import numpy as np
from start_tests import run_experiment_lib


class DT_lib:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def evaluate_min_samples_leaf(self, min_samples_leaf_values=range(1, 21)):
        hyperparam_name = 'min_samples_leaf'
        fixed_params = {
            'max_depth': None,
            'criterion': 'entropy',
            'min_samples_split': 2,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, min_samples_leaf_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params)

    def evaluate_min_samples_split(self, min_samples_split_values=range(2, 21)):
        hyperparam_name = 'min_samples_split'
        fixed_params = {
            'max_depth': None,
            'criterion': 'gini',
            'min_samples_leaf': 1,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, min_samples_split_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params)

    def evaluate_max_leaf_nodes(self, max_leaf_nodes_values=None):
        if max_leaf_nodes_values is None:
            max_leaf_nodes_values = [None, 5, 10, 20, 50, 100]

        hyperparam_name = 'max_leaf_nodes'
        fixed_params = {
            'max_depth': None,
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, max_leaf_nodes_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params)

    def evaluate_min_impurity_decrease(self, min_impurity_decrease_values=np.linspace(0.0, 0.15, 11)):
        hyperparam_name = 'min_impurity_decrease'
        fixed_params = {
            'max_depth': None,
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, min_impurity_decrease_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params)

    def evaluate_ccp_alpha(self, ccp_alpha_values=np.linspace(0.0, 0.05, 30)):
        hyperparam_name = 'ccp_alpha'
        fixed_params = {
            'max_depth': None,
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, ccp_alpha_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params)

    def evaluate_max_depth(self, max_depth_values=None):
        if max_depth_values is None:
            max_depth_values = list(range(1, 21)) + [None]

        hyperparam_name = 'max_depth'
        fixed_params = {
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        run_experiment_lib(hyperparam_name, max_depth_values, self.X_train, self.y_train, self.X_val, self.y_val, fixed_params, plot_type="accuracy")
