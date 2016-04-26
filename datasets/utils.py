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
    domain_data = {
            'X_train': [data['X_train'] for data in datasets],
            'X_val': [data['X_val'] for data in datasets],
            'X_test': [data['X_test'] for data in datasets],
            'y_train': None,
            'y_val': None,
            'y_test': None,
            'batchsize': batchsize,
            }

    return domain_data


if __name__ == '__main__':
    a = np.arange(20).reshape(-1, 2)
    b = np.arange(20).reshape(-1, 2)
    c = np.arange(20).reshape(-1, 2)

    print(*shuffle_array(a))
    print(*shuffle_array(a, b))
    print(*shuffle_array(a, b, c))
