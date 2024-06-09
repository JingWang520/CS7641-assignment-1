import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
import random

random.seed(42)

# Loading and pre-processing data
data = pd.read_csv('../data/mobile_train.csv')
X = data.drop('price_range', axis=1)  # 目标变量是 'price_range'
y = data['price_range']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Defining classifiers and hyperparameter grids
classifiers = {
    'AdaBoost': AdaBoostClassifier(),
}

param_grids = {
    'AdaBoost': {
        'learning_rate': [0.1, 0.5,0.7,0.85, 1,1.2],
        'n_estimators': list(range(10, 100, 5)),
    },
}

scorer = make_scorer(accuracy_score)

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
    # plt.grid()
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cbar_kws={'label': 'Mean Test Score'})
    plt.title(f"Validation Accuracy Heatmap for {name}")
    plt.xlabel(param_2)
    plt.ylabel(param_1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../result/images/mobile/{name}_heatmap_add.png')
    plt.show()

    return grid_search.best_params_

best_params = {}
for name in classifiers:
    best_params[name] = train_and_plot_heatmap(name, classifiers[name], param_grids[name])

