from build_data import get_processed_data
from dt import DTclassifier_custom
from dt_lib import DT_lib
from start_tests import run_random_forest_experiment_custom, run_random_forest_experiment_lib, \
    run_gradient_boosting_experiment_lib


def main():
    X_train_np, y_train_np, X_val_np, y_val_np = get_processed_data()

    DTcustom = DTclassifier_custom(max_depth=None, criterion='entropy', min_samples_split=4, min_samples_leaf=2)

    DTcustom.evaluate_with_min_samples_leaf(X_train_np, y_train_np, X_val_np, y_val_np)
    DTcustom.evaluate_min_samples_split(X_train_np, y_train_np, X_val_np, y_val_np)
    DTcustom.evaluate_with_criterion(X_train_np, y_train_np, X_val_np, y_val_np)
    DTcustom.evaluate_with_max_depth(X_train_np, y_train_np, X_val_np, y_val_np)

    DTlib = DT_lib(X_train_np, y_train_np, X_val_np, y_val_np)
    DTlib.evaluate_min_samples_split()
    DTlib.evaluate_min_samples_leaf()
    DTlib.evaluate_min_impurity_decrease()
    DTlib.evaluate_max_leaf_nodes()
    DTlib.evaluate_ccp_alpha()
    DTlib.evaluate_max_depth()

    n_estimators_values = [5, 10, 20, 50, 100, 150]
    run_random_forest_experiment_custom(n_estimators_values, X_train_np, y_train_np, X_val_np, y_val_np)
    run_random_forest_experiment_lib(n_estimators_values, X_train_np, y_train_np, X_val_np, y_val_np)

    run_gradient_boosting_experiment_lib(n_estimators_values, X_train_np, y_train_np, X_val_np, y_val_np)


if __name__ == "__main__":
    main()
