"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q1"""
        distances = euclidean_dist_squared(X_hat, self.X)
        sorted_args = np.argsort(distances)[:,0:self.k]
        votes = self.y[sorted_args]


        return np.array([utils.mode(row) for row in votes])

