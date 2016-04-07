#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from sklearn.preprocessing import Normalizer
np.random.seed(12345)


def shuffle_array(*args):
    """
    Shuffle the given data. Keeps the relative associations arr_j[i] <-> arr_k[i].

    Params
    ------
        args: (numpy arrays tuple) arr_1, arr_2, ..., arr_n to be shuffled.
    Return
    ------
        X, y : the shuffled arrays.
    """
    # Assert that there is at least one array
    if len(args) == 0:
        raise ValueError('shuffle must take at least one array')
    length = args[0].shape[0]
    # Assert that every array have the same 1st dimension length:
    for i, arr in enumerate(args):
        assert arr.shape[0] == length, "Every array should have the same shape: " \
                        " array {} length = {}  array 1 length = {} ".format(i+1, arr.shape[0], length)
    # Make the random indices
    indices = np.arange(length)
    np.random.shuffle(indices)
    # Return shuffled arrays
    return tuple(arr[indices] for arr in args)


def domain_X_y(X_list, shuffle=True):
    """
    Build the domain dataset.

    Params
    ------
        X_list: (numpy arrays tuple) X_1, X_2, ..., X_n 
            comming from different domains to be stacked.
        shuffle: (default=True) if True the domain instances will be shuffled.

    Return
    ------
        X: the stacked domain data
        y: the corresponding domain label (numpy 1D array of labels in [0..n]).
    
    """
    X = np.vstack(X_list)

    y = np.hstack([i*np.ones((arr.shape[0],), dtype=np.int32)
               for i, arr in enumerate(X_list)])
    if shuffle:
        X, y = shuffle_array(X, y)
    return X, y


def make_domain_dataset(datasets):
    """
    Make a domain datasets out of the given datasets.

    Params
    ------
        datasets: a list of datasets (dicts with the separated data)

    Return
    ------
        domain_data: dict with the separated data
    """
    X_train, y_train = domain_X_y([t['X_train'] for t in datasets])
    X_val, y_val = domain_X_y([t['X_val'] for t in datasets])
    X_test, y_test = domain_X_y([t['X_test'] for t in datasets])
    batchsize = sum([t['batchsize'] for t in datasets])
    domain_data = {
                'X_train': X_train,
                'y_train': y_train, 
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'batchsize':batchsize,
                }

    return domain_data


def diag_dominant_matrix(size):
    """
    https://matthewhr.wordpress.com/2013/09/01/how-to-construct-an-invertible-matrix-just-choose-large-diagonals/
    """
    matrix = np.random.random((size, size))
    s = np.sum(matrix, axis=1)
    matrix += np.eye(size) * s
    eigens = np.linalg.eig(matrix)[0]
    if (eigens == 0).any():
        raise ValueError('The matrix is not invertible. Internet lied to me ! {}'.format(str(eigens)))
    return matrix


def diag_dataset(source_data, normalize=False):
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

    A = diag_dominant_matrix(size)
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

    domain_data = make_domain_dataset([source_data, target_data])
    return source_data, target_data, domain_data


def random_mat_dataset(source_data, normalize=False):
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

    A = np.random.random((size, size))
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

    domain_data = make_domain_dataset([source_data, target_data])
    return source_data, target_data, domain_data


def rotate_data(X, angle=35.):
    """Apply a rotation on a 2D dataset.
    """
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    X_r = np.empty_like(X)
    X_r[:] = X[:].dot(rotMatrix)
    return X_r


if __name__ == '__main__':
    a = np.arange(20).reshape(-1, 2)
    b = np.arange(20).reshape(-1, 2)
    c = np.arange(20).reshape(-1, 2)

    print(*shuffle_array(a))
    print(*shuffle_array(a, b))
    print(*shuffle_array(a, b, c))
