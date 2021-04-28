# Author: Kshitij Kayastha
# Date: April 25, 2021


import numpy as np
from .validation import is_column, is_row

def convert_to_column(y):
    """
    Convert a 1-dimensional array to a column

    Parameters
    ----------
    y : array_like of shape (n_samples,) or (n_samples, 1) or (1, n_samples)
        Input array

    Returns
    -------
    column : array_like of shape (n_samples, 1)
        Column array
    """
    y = np.asarray(y)
    if is_column(y):
        return y
    elif is_row(y):
        column = np.array([[elem] for elem in y])
        return column
    else:
        raise ValueError(
            "y should be a 1d array, "
            "got an array of shape {} instead.".format(y.shape))


def add_bias(X, axis=0, first=True):
    """
    Add a bias column/row to an array

    Parameters
    ----------
    X : array_like of shape (n_samples, m_features)
        Input array
    axis : int {0: vertical, 1: horizontal}, default 0
        Axis along which the bias will be added
    first: bool, default True
        Add bias to the first column/row. If false bias
        will be added to the last column/row
    """
    X = np.asarray(X)
    shape = X.shape
    if axis == 0:
        bias = np.ones((shape[0], 1))
        if first:
            X = np.hstack((bias, X))
        else:
            X = np.hstack((X, bias))
    elif axis == 1:
        bias = np.ones((1,shape[1]))
        if first:
            X = np.vstack((bias, X))
        else:
            X = np.vstack((X, bias))
    
    return X 

def mean_variance_axis(X, axis=0):
    X = np.asarray(X)
    mu = np.mean(X, axis=axis)
    sigma_sq = np.var(X, axis=axis, ddof=1)

    return mu, sigma_sq 

