# Author: Kshitij Kayastha
# Date: April 28, 2021


import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

class PCA:
    """
    Principal Component Analysis

    Dimensionality reduction of the data to project it to a lower dimensional space.
    The input data is scaled for each feature.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If n_components is not set all components are kept

    Attributes
    ----------
    eig_vals : array_like of shape (1, n_features)
        Eigenvalues of the covariance matrix
    eig_vecs : array_like of shape (n_features, n_feautures)
        Eigenvectors corresponding to the eigenvalues of the covariance matrix
    components: array_like of shape (n_samples, n_components)
        Principal Components
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self._fitted = False

    def fit(self, X):
        """
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training samples

        Returns
        -------
        self : returns an instance of self.
        """
        X = np.asarray(X)
        if self.n_components is None:
            self.n_components = X.shape[1]
        X = scale(X)
        sigma = np.cov(X.T)
        sigma_shape = sigma.shape
        self.eig_vals, self.eig_vecs = np.linalg.eig(sigma)
        
        sort_key = np.argsort(self.eig_vals)[::-1]
        self.eig_vals = self.eig_vals[sort_key]
        self.eig_vecs = self.eig_vecs[:, sort_key]
        
        self.explained_variances = np.array(
            [self.eig_vals[i] / np.sum(self.eig_vals) for i in range(sigma_shape[0])])

        self._fitted = True

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training samples

        Returns
        -------
        components : array_like of shape (n_samples, n_components)
            Transformed values
        """
        if not self._fitted:
            raise ValueError('This PCA instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.')
        X = np.asarray(X)
        self.components = np.dot(X, self.eig_vecs[:, :self.n_components])
        return self.components

    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training samples

        Returns
        -------
        components : array_like of shape (n_samples, n_components)
            Transformed values
        """
        self.fit(X)
        return self.transform(X)