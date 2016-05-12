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


def diag_dataset(source_data, normalize=False):
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
    X_train = source_data['X_train']
    y_train = source_data['y_train']
    X_val = source_data['X_val']
    y_val = source_data['y_val']
    X_test = source_data['X_test']
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']
    size = np.prod(X_train.shape[1:])

    y_t_train = y_train
    y_t_val = y_val
    y_t_test = y_test

    A = _diag_dominant_matrix(size)
    if normalize:
        normalizer = Normalizer()
        X_t_train = normalizer.fit_transform(np.dot(X_train.reshape(-1, size), A)).reshape(X_train.shape)
        X_t_val = normalizer.fit_transform(np.dot(X_val.reshape(-1, size), A)).reshape(X_val.shape)
        X_t_test = normalizer.fit_transform(np.dot(X_test.reshape(-1, size), A)).reshape(X_test.shape)
    else:
        X_t_train = np.dot(X_train.reshape(-1, size), A).reshape(X_train.shape)
        X_t_val = np.dot(X_val.reshape(-1, size), A).reshape(X_val.shape)
        X_t_test = np.dot(X_test.reshape(-1, size), A).reshape(X_test.shape)
        
    target_data = {
                'X_train': X_t_train,
                'y_train': y_t_train,
                'X_val': X_t_val,
                'y_val': y_t_val,
                'X_test': X_t_test,
                'y_test': y_t_test,
                'batchsize': batchsize,
                }

    return target_data


def random_mat_dataset(source_data, normalize=False, random_state=None):
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

    X_train = source_data['X_train']
    y_train = source_data['y_train']
    X_val = source_data['X_val']
    y_val = source_data['y_val']
    X_test = source_data['X_test']
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']
    size = np.prod(X_train.shape[1:])

    y_t_train = y_train
    y_t_val = y_val
    y_t_test = y_test

    rand_state = np.random.RandomState(random_state)
    A = rand_state.random((size, size))
    if normalize:
        normalizer = Normalizer()
        X_t_train = normalizer.fit_transform(np.dot(X_train.reshape(-1, size), A)).reshape(X_train.shape)
        X_t_val = normalizer.fit_transform(np.dot(X_val.reshape(-1, size), A)).reshape(X_val.shape)
        X_t_test = normalizer.fit_transform(np.dot(X_test.reshape(-1, size), A)).reshape(X_test.shape)
    else:
        X_t_train = np.dot(X_train.reshape(-1, size), A).reshape(X_train.shape)
        X_t_val = np.dot(X_val.reshape(-1, size), A).reshape(X_val.shape)
        X_t_test = np.dot(X_test.reshape(-1, size), A).reshape(X_test.shape)

    target_data = {
                'X_train': X_t_train,
                'y_train': y_t_train,
                'X_val': X_t_val,
                'y_val': y_t_val,
                'X_test': X_t_test,
                'y_test': y_t_test,
                'batchsize': batchsize,
                }

    return target_data

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


def rotate_dataset(source_data, angle=35.):
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

    X_train = source_data['X_train']
    y_train = source_data['y_train']
    X_val = source_data['X_val']
    y_val = source_data['y_val']
    X_test = source_data['X_test']
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']

    target_data = {
                'X_train': _rotate_data(X_train, angle=angle),
                'y_train': y_train,
                'X_val': _rotate_data(X_val, angle=angle),
                'y_val': y_val,
                'X_test': _rotate_data(X_test, angle=angle),
                'y_test': y_test,
                'batchsize': batchsize,
                }

    return target_data


# ============================================================================
#                   Invertions
# ============================================================================

def invert_dataset(source_data, pivot=1):
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

    X_train = source_data['X_train']
    y_train = source_data['y_train']
    X_val = source_data['X_val']
    y_val = source_data['y_val']
    X_test = source_data['X_test']
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']
    size = np.prod(X_train.shape[1:])

    target_data = {
                'X_train': (pivot-X_train),
                'y_train': y_train,
                'X_val': (pivot-X_val),
                'y_val': y_val,
                'X_test': (pivot-X_test),
                'y_test': y_test,
                'batchsize': batchsize,
                }

    return target_data

# ============================================================================
#                   Permutations
# ============================================================================

def mirror_dataset(source_data, shape=(-1,28,28)):
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

    X_train = source_data['X_train']
    y_train = source_data['y_train']
    X_val = source_data['X_val']
    y_val = source_data['y_val']
    X_test = source_data['X_test']
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']

    target_data = {
                'X_train': np.fliplr(X_train.reshape(shape)),
                'y_train': y_train,
                'X_val': np.fliplr(X_val.reshape(shape)),
                'y_val': y_val,
                'X_test': np.fliplr(X_test.reshape(shape)),
                'y_test': y_test,
                'batchsize': batchsize,
                }

    return target_data


def random_permut_dataset(source_data, random_state=None):
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

    X_train = np.copy(source_data['X_train'])
    y_train = source_data['y_train']
    X_val = np.copy(source_data['X_val'])
    y_val = source_data['y_val']
    X_test = np.copy(source_data['X_test'])
    y_test = source_data['y_test']
    batchsize = source_data['batchsize']

    # Take care of 3D or n-D data
    rand_state = np.random.RandomState(random_state)
    permutation = rand_state.permutation(np.prod(X_train.shape[1:]))
    
    shape = X_train.shape
    X_train = X_train.reshape(-1, np.prod(shape[1:]))
    X_train = X_train[:, permutation].reshape(shape)

    shape = X_val.shape
    X_val = X_val.reshape(-1, np.prod(shape[1:]))
    X_val = X_val[:, permutation].reshape(shape)

    shape = X_test.shape
    X_test = X_test.reshape(-1, np.prod(shape[1:]))
    X_test = X_test[:, permutation].reshape(shape)

    target_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'batchsize': batchsize,
                }

    return target_data
