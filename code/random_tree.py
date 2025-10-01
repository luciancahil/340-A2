from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []


    def fit(self, X, y):
        for i in range(self.num_trees):
            new_tree = RandomTree(self.max_depth)

            new_tree.fit(X, y)
            self.trees.append(new_tree)


    def predict(self, X_pred):
        preds = np.array([tree.predict(X_pred) for tree in self.trees]).T
        preds = np.array([utils.mode(row) for row in preds])
        return preds

