#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

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

    y = np.hstack([i*np.ones((X.shape[0],), dtype=np.int32), 
               for i, X in enumerate(X_list)])
    if shuffle:
        X, y = shuffle_array(X, y)
    return X, y

if __name__ == '__main__':
    a = np.arange(10)
    b = np.arange(10)
    c = np.arange(10)

    print(*shuffle_array(a))
    print(*shuffle_array(a, b))
    print(*shuffle_array(a, b, c))
