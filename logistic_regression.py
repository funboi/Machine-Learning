# Author: Kshitij Kayastha
# Date: April 28, 2021


import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from utils.transform import add_bias, convert_to_column, mean_variance_axis
from sklearn.preprocessing import scale

class LogisticRegression:
    """
    Logistic Regression classifier

    Parameters
    ----------
    eta : float, default=0.05
        Learning rate
    epochs : int, default=10000
        Number of iterations over the training set
    scale : bool, default True
        Scale the training samples

    Attributes
    ----------
    theta : weights after fitting the model
    cost : total error of the model after each iteration
    epsilon: error offset
    """
    def __init__(self, eta=0.05, epochs=1000, scale=True):
        self.scale = scale
        self.eta = eta
        self.epochs = epochs
        self._epsilon = 1 + 10**(-8)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training samples
        y : array_like of shape (n_samples, 1)
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        X = np.asarray(X)
        y = convert_to_column(y)

        if self.scale:
            X = scale(X)
        X = add_bias(X)
        n_samples = X.shape[0]
        m_features = X.shape[1]
        
        self.theta = np.zeros((m_features, 1))
        self.mean, self.variance = mean_variance_axis(X, axis=0)
        self.cost = []

        for _ in range(self.epochs):
            y_pred = self._sigmoid(np.dot(X, self.theta))
            residuals = y - y_pred
            self.theta += (1 / n_samples) * self.eta * np.dot(X.T, residuals)
            self.cost.append(self._cost(y, y_pred))

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array_like
            Test samples

        Returns
        -------
        y_pred : array_like
            Predicted values
        """
        X = np.asarray(X)
        if self.scale:
            X = scale(X)
        X = add_bias(X)

        z = self._sigmoid(np.dot(X, self.theta))
        y_pred = np.array([1 if val >= 0.5 else 0 for val in z])

        return y_pred

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def _cost(self, y_true, y_pred):
        total_cost = -(1 / len(y_true)) * np.sum(
            y_true * np.log(y_pred+self._epsilon) +
            (1 - y_true) * np.log(1 - y_pred + self._epsilon)
        )
        return total_cost