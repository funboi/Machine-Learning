import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from utils.transform import convert_to_column, add_bias
from sklearn.preprocessing import scale

class LinearRegression:

    """
    Oridinary least squares Linear Regression (closed form)

    Parameters
    ----------
    eta : float
        Learning rate
    n_iters : int
        Number of iterations over the training set

    Attributes
    ----------
    theta : weights after fitting the model
    cost : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iters=1000):
        self.eta = eta
        self.n_iters = n_iters
        self.theta = None
        self.cost = []


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
        m = X.shape[0]
        
        if issparse(X):
            raise ValueError("Linear regression on sparse matrices has not yet been implemented")
            
        X = scale(X)
        X = add_bias(X)

        symm = np.matmul(X.T, X)
        symm_inv = np.linalg.pinv(symm)
        symm_inv_X = np.matmul(symm_inv, X.T)
        self.theta = np.matmul(symm_inv_X, y)

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
        X = scale(X)
        X = add_bias(X)
        
        y_pred = np.dot(X, self.theta)
        return y_pred
