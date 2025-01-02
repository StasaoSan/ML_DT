import matplotlib.pyplot as plt

def plot_tree_depth(hyperparam_name, hyperparam_values, tree_depths, custom_realize):
    plt.figure(figsize=(10, 6))
    plt.plot(hyperparam_values, tree_depths, marker='o')
    plt.xlabel(hyperparam_name)
    plt.ylabel('Tree Depth')
    if custom_realize:
        plt.title(f'Tree depth from: {hyperparam_name} (custom realization)')
    else:
        plt.title(f'Tree depth from: {hyperparam_name} (lib realization)')
    plt.grid(True)
    plt.show()

def plot_accuracy(hyperparam_name, hyperparam_values, train_accuracies, val_accuracies, custom_realize):
    plt.figure(figsize=(10, 6))
    plt.plot(hyperparam_values, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(hyperparam_values, val_accuracies, marker='s', label='Validation Accuracy')
    plt.xlabel(hyperparam_name)
    plt.ylabel('Accuracy')
    if (custom_realize):
        plt.title(f'Accuracy from: {hyperparam_name} (custom realization)')
    else:
        plt.title(f'Accuracy from: {hyperparam_name} (lib realization)')
    plt.legend()
    plt.grid(True)
    plt.show()
