# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from sklearn.preprocessing import Normalizer
np.random.seed(12345)

# ============================================================================
#                   Matrix products
# ============================================================================

def _diag_dominant_matrix(size, safe=False):
    """
    https://matthewhr.wordpress.com/2013/09/01/how-to-construct-an-invertible-matrix-just-choose-large-diagonals/
    """
    matrix = np.random.random((size, size))
    s = np.sum(matrix, axis=1)
    matrix += np.eye(size) * s
    if safe:
        eigens = np.linalg.eig(matrix)[0]
        if (eigens == 0).any():
            raise ValueError('The matrix is not invertible. Internet lied to me ! {}'.format(str(eigens)))
    return matrix


def diag_dominant(X, y, normalize=False):
    """
    Transform the given dataset by applying a diagonal dominant matrix to it.

    target_data <- source_data . Matrix
    
    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """
    size = np.prod(X.shape[1:])
    y_t = np.copy(y)
    
    A = _diag_dominant_matrix(size)
    if normalize:
        normalizer = Normalizer()
        X_t = normalizer.fit_transform(np.dot(X.reshape(-1, size), A)).reshape(X.shape)
    else:
        X_t = np.dot(X.reshape(-1, size), A).reshape(X.shape)
        
    return X_t, y_t


def random_mat(X, y, normalize=False, random_state=None):
    """
    Transform the given dataset by applying a random matrix to it.

    target_data <- source_data . Matrix
    
    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """

    size = np.prod(X.shape[1:])
    y_t = np.copy(y)

    rand_state = np.random.RandomState(random_state)
    A = rand_state.rand(size, size)
    if normalize:
        normalizer = Normalizer()
        X_t = normalizer.fit_transform(np.dot(X.reshape(-1, size), A)).reshape(X.shape)
    else:
        X_t = np.dot(X.reshape(-1, size), A).reshape(X.shape)
    return X_t, y_t

# ============================================================================
#                   Rotations
# ============================================================================

def _rot_mat(angle):
    theta = (angle/180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    return rot_matrix

 
def _rotate_data(X, angle=35.):
    """Apply a rotation on a 2D dataset.
    """
    theta = (angle/180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    X_r = np.empty_like(X)
    X_r[:] = X[:].dot(rot_matrix)
    return X_r


def rotate(X, y, angle=35.):
    """
    Transform the given dataset by applying a rotation to it.

    target_data <- source_data . Rotation_Matrix

    Can be used only on 2D datasets !

    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """

    y_t = np.copy(y)
    X_t =  _rotate_data(X, angle=angle)
    return X_t, y_t


# ============================================================================
#                   Invertions
# ============================================================================

def invert(X, y, pivot=1):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- (pivot - source_data)
    
    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """
    y_t = np.copy(y)
    X_t = pivot - X
    return X_t, y_t

    
# ============================================================================
#                   Permutations
# ============================================================================

def mirror(X, y, shape=(-1,28,28)):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- np.fliplr(source_data.reshape(shape))

    TODO : This function works only with MNIST for now !!!
    
    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """

    X_t =  np.fliplr(np.copy(X).reshape(shape))
    y_t = np.copy(y)
    return X_t, y_t


def random_permut(X, y, random_state=None):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- random_permutation of feature(source_data)

    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """
    y_t = np.copy(y)

    X_t = np.copy(X)
    # Take care of 3D or n-D data
    rand_state = np.random.RandomState(random_state)
    permutation = rand_state.permutation(np.prod(X_t.shape[1:]))
    
    shape = X_t.shape
    X_t = X_t.reshape(-1, np.prod(shape[1:]))
    X_t = X_t[:, permutation].reshape(shape)
    return X_t, y_t


# ============================================================================
#                   Apply a given function
# ============================================================================

def apply_fun(source_data, fun):
    """
    Transform the given dataset by applying an operation to it.

    target_data <- fun(source_data)

    Params
    ------
        source_data: a dataset (dict with the separated data)

    Return
    ------
        target_data: dict with the separated transformed data

    """

    X_train = np.copy(source_data['X_train'])
    y_train = source_data['y_train']
    X_val = np.copy(source_data['X_val'])
    y_val = source_data['y_val']
    X_test = np.copy(source_data['X_test'])
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']

    target_data = {
                'X_train': fun(X_train),
                'y_train': y_train,
                'X_val': fun(X_val),
                'y_val': y_val,
                'X_test': fun(X_test),
                'y_test': y_test,
                'batchsize': batchsize,
                }

    return target_data

