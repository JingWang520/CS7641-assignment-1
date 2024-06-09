import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time
import random

random.seed(42)

# Loading and pre-processing data
data = pd.read_csv('../data/heart_statlog_cleveland_hungary_final.csv')
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# define the classifiers and hyperparameter grids
classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'NeuralNetwork': MLPClassifier(max_iter=1000),
    'AdaBoost': AdaBoostClassifier(),
    'SVM': SVC(probability=True),
    'kNN': KNeighborsClassifier()
}

param_grids = {
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1, 60, 4))
    },
    'NeuralNetwork': {
        'alpha': [0.0001, 0.001],
        'hidden_layer_sizes': [(10,20), (20,20), (30,20), (40, 20), (50,20), (60,20), (70,10), (80,10), (90,10), (100,10), (10,10)],
    },
    'AdaBoost': {
        'learning_rate': [0.1, 0.5, 1],
        'n_estimators': list(range(10, 110, 10)),
    },
    'SVM': {
        'gamma': [0.0001, 0.001, 0.01, 0.1],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    'kNN': {
        'weights': ['uniform', 'distance'],
        'n_neighbors': list(range(2, 100, 10)),
    }
}

# custom accuracy_score scoring function
scorer = make_scorer(accuracy_score)

# Hyperparameter tuning using grid search and cross-validation and calculating accuracy
def train_and_plot_heatmap(name, classifier, param_grid):
    print(f"Training {name}...")
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    results = pd.DataFrame(grid_search.cv_results_)

    params = results["params"]
    scores = results["mean_test_score"]

    param_1, param_2 = list(param_grid.keys())
    param_grid = pd.DataFrame([dict(**p, mean_test_score=s) for p, s in zip(params, scores)])

    pivot_table = param_grid.pivot_table(values='mean_test_score', index=param_1, columns=param_2, aggfunc=np.mean)

    plt.figure(figsize=(10, 8))

    sns.heatmap(pivot_table, annot=True, fmt=".3f", cbar_kws={'label': 'Mean Test Score'})
    plt.title(f"Validation Accuracy Heatmap for {name}")
    plt.xlabel(param_2)
    plt.ylabel(param_1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../result/images/heart/{name}_heatmap.png')
    plt.show()

    return grid_search.best_params_

best_params = {}
for name in classifiers:
    best_params[name] = train_and_plot_heatmap(name, classifiers[name], param_grids[name])

# plot
def plot_learning_curve(estimator, title, X_train, y_train, X_test, y_test, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)


    estimator.fit(X_train, y_train)

    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'../result/images/heart/{title}.png')
    return plt


def plot_accuracy_vs_epoch(estimator, X_train, y_train, X_test, y_test, title, epochs=100):
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        estimator.max_iter = epoch
        estimator.fit(X_train, y_train)
        train_accuracies.append(accuracy_score(y_train, estimator.predict(X_train)))
        test_accuracies.append(accuracy_score(y_test, estimator.predict(X_test)))

    plt.figure()
    plt.grid()
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(f'../result/images/heart/{title}.png')
    plt.show()


best_nn = MLPClassifier(**best_params['NeuralNetwork'], max_iter=1, warm_start=True)
plot_accuracy_vs_epoch(best_nn, X_train, y_train, X_test, y_test, "Accuracy vs Epoch (Neural Network)")

best_ada = AdaBoostClassifier(**best_params['AdaBoost'])
plot_learning_curve(best_ada, "Learning Curve (AdaBoost)", X_train, y_train, X_test, y_test, cv=5, n_jobs=-1)
plt.show()

best_dt = DecisionTreeClassifier(**best_params['DecisionTree'])
plot_learning_curve(best_dt, "Learning Curve (Decision Tree)", X_train, y_train, X_test, y_test, cv=5, n_jobs=-1)
plt.show()

best_svm = SVC(**best_params['SVM'], probability=True)
plot_learning_curve(best_svm, "Learning Curve (SVM)", X_train, y_train, X_test, y_test, cv=5, n_jobs=-1)
plt.show()

best_knn = KNeighborsClassifier(**best_params['kNN'])
plot_learning_curve(best_knn, "Learning Curve (kNN)", X_train, y_train, X_test, y_test, cv=5, n_jobs=-1)
plt.show()

# Evaluating the best models
best_nn.fit(X_train, y_train)
nn_test_accuracy = accuracy_score(y_test, best_nn.predict(X_test))
print(f"Neural Network Test Accuracy: {nn_test_accuracy:.3f}")

best_ada.fit(X_train, y_train)
ada_test_accuracy = accuracy_score(y_test, best_ada.predict(X_test))
print(f"AdaBoost Test Accuracy: {ada_test_accuracy:.3f}")

best_dt.fit(X_train, y_train)
dt_test_accuracy = accuracy_score(y_test, best_dt.predict(X_test))
print(f"Decision Tree Test Accuracy: {dt_test_accuracy:.3f}")

best_svm.fit(X_train, y_train)
svm_test_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
print(f"SVM Test Accuracy: {svm_test_accuracy:.3f}")

best_knn.fit(X_train, y_train)
knn_test_accuracy = accuracy_score(y_test, best_knn.predict(X_test))
print(f"kNN Test Accuracy: {knn_test_accuracy:.3f}")


model_names = ['Neural Network', 'AdaBoost', 'Decision Tree', 'SVM', 'kNN']
accuracies = []
training_times = []

# Train and evaluate each model while recording training time
start_time = time.time()
best_nn.fit(X_train, y_train)
nn_test_accuracy = accuracy_score(y_test, best_nn.predict(X_test))
accuracies.append(nn_test_accuracy)
training_times.append(time.time() - start_time)

start_time = time.time()
best_ada.fit(X_train, y_train)
ada_test_accuracy = accuracy_score(y_test, best_ada.predict(X_test))
accuracies.append(ada_test_accuracy)
training_times.append(time.time() - start_time)

start_time = time.time()
best_dt.fit(X_train, y_train)
dt_test_accuracy = accuracy_score(y_test, best_dt.predict(X_test))
accuracies.append(dt_test_accuracy)
training_times.append(time.time() - start_time)

start_time = time.time()
best_svm.fit(X_train, y_train)
svm_test_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
accuracies.append(svm_test_accuracy)
training_times.append(time.time() - start_time)

start_time = time.time()
best_knn.fit(X_train, y_train)
knn_test_accuracy = accuracy_score(y_test, best_knn.predict(X_test))
accuracies.append(knn_test_accuracy)
training_times.append(time.time() - start_time)

# Plotting a histogram of the best accuracy of the model
plt.figure(figsize=(12, 6))

plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Best Test Accuracy of Each Model')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('../result/images/heart/model_accuracies.png')
plt.show()

# Plotting a histogram of model training time
plt.figure(figsize=(12, 6))
plt.bar(model_names, training_times, color='salmon')
plt.xlabel('Models')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time of Each Model')
for i, v in enumerate(training_times):
    plt.text(i, v + 0.01, f"{v:.2f}s", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('../result/images/heart/model_training_times.png')
plt.show()
