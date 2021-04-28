# Author: Kshitij Kayastha
# Date: April 25, 2021


import numpy as np

def is_1d(y):
    """
    Checks if input array is 1-dimensional

    Parameters
    ----------
    y : array_like
        Input array
    """
    y = np.asarray(y)
    shape = y.shape
    if len(shape) == 1:
        return True
    if shape[0] == 1 or shape[1] == 1:
        return True
    return False

def is_row(y):
    """
    Checks if input array is a row

    Parameters
    ----------
    y : array_like
        Input array
    """
    y = np.asarray(y)
    shape = y.shape
    if len(shape) == 1:
        return True
    return shape[0] == 1

def is_column(y):
    """
    Checks if input array is a column

    Parameters
    ----------
    y : array_like
        Input array
    """
    y = np.asarray(y)
    shape = y.shape
    if len(shape) == 1:
        return False
    return shape[1] == 1
    