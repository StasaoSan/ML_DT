from drawing_plots import plot_tree_depth, plot_accuracy


def run_experiment_custom(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params, plot_type="tree_depth"):
    from dt import DTclassifier_custom

    tree_depths = []
    train_accuracies = []
    val_accuracies = []

    for value in hyperparam_values:
        params = fixed_params.copy()
        params[hyperparam_name] = value

        custom_tree = DTclassifier_custom(**params)
        custom_tree.fit(X_train, y_train)

        depth = custom_tree.get_tree_depth()
        tree_depths.append(depth)

        y_train_pred = custom_tree.predict(X_train)
        y_val_pred = custom_tree.predict(X_val)

        train_accuracy = DTclassifier_custom.calculate_accuracy(y_train, y_train_pred)
        val_accuracy = DTclassifier_custom.calculate_accuracy(y_val, y_val_pred)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"{hyperparam_name}: {value}, Tree Depth: {depth}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    if plot_type == "tree_depth":
        plot_tree_depth(hyperparam_name, hyperparam_values, tree_depths,  custom_realize=True)
    elif plot_type == "accuracy":
        plot_accuracy(hyperparam_name, hyperparam_values, train_accuracies, val_accuracies, custom_realize=True)


def run_experiment_lib(hyperparam_name, hyperparam_values, X_train, y_train, X_val, y_val, fixed_params, plot_type="tree_depth"):
    from sklearn.tree import DecisionTreeClassifier

    tree_depths = []
    train_accuracies = []
    val_accuracies = []

    for value in hyperparam_values:
        params = fixed_params.copy()
        params[hyperparam_name] = value

        lib_tree = DecisionTreeClassifier(**params)
        lib_tree.fit(X_train, y_train)

        depth = lib_tree.get_depth()
        tree_depths.append(depth)

        train_accuracy = lib_tree.score(X_train, y_train)
        val_accuracy = lib_tree.score(X_val, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"{hyperparam_name}: {value}, Tree Depth: {depth}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    if plot_type == "tree_depth":
        plot_tree_depth(hyperparam_name, hyperparam_values, tree_depths, custom_realize=False)
    elif plot_type == "accuracy":
        plot_accuracy(hyperparam_name, hyperparam_values, train_accuracies, val_accuracies, custom_realize=False)


def run_random_forest_experiment_custom(n_estimators_values, X_train, y_train, X_val, y_val):
    from dt import DTclassifier_custom
    from random_forest_custom import RandomForestClassifierCustom

    train_accuracies = []
    val_accuracies = []

    for n_estimators in n_estimators_values:
        rf_custom = RandomForestClassifierCustom(n_estimators=n_estimators, random_state=42)
        rf_custom.fit(X_train, y_train)

        y_train_pred = rf_custom.predict(X_train)
        y_val_pred = rf_custom.predict(X_val)

        train_accuracy = DTclassifier_custom.calculate_accuracy(y_train, y_train_pred)
        val_accuracy = DTclassifier_custom.calculate_accuracy(y_val, y_val_pred)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"n_estimators (rf_custom): {n_estimators}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    plot_accuracy('n_estimators (rf)', n_estimators_values, train_accuracies, val_accuracies, custom_realize=True)


def run_random_forest_experiment_lib(n_estimators_values, X_train, y_train, X_val, y_val):
    from sklearn.ensemble import RandomForestClassifier

    train_accuracies = []
    val_accuracies = []

    for n_estimators in n_estimators_values:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)

        train_accuracy = rf.score(X_train, y_train)
        val_accuracy = rf.score(X_val, y_val)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"n_estimators (rf_lib): {n_estimators}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    plot_accuracy('n_estimators (rf)', n_estimators_values, train_accuracies, val_accuracies, custom_realize=False)


def run_gradient_boosting_experiment_lib(n_estimators_values, X_train, y_train, X_val, y_val):
    from sklearn.ensemble import GradientBoostingClassifier

    train_accuracies = []
    val_accuracies = []

    for n_estimators in n_estimators_values:
        gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)

        train_accuracy = gb.score(X_train, y_train)
        val_accuracy = gb.score(X_val, y_val)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"n_estimators (gradient boosting): {n_estimators}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    plot_accuracy('n_estimators (gradient boosting)', n_estimators_values, train_accuracies, val_accuracies, custom_realize=False)
