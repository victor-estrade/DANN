# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from sklearn.preprocessing import Normalizer
np.random.seed(12345)

# this class is what I use for every thing when I need to gather several 
# things at the same place.
# Yes ! I do not care at all that it is dirty coding !
class AttributeDict(dict):
    """
    A dictionnary that allow accessing and setting its elements
    like in an object
    
    Example:
    >>> d = AttributeDict({'a':5, 'b':2})
    >>> print(d.a == 5)
    True
    >>> print(d.b == 5)
    False
    >>> d.something = 'foo'
    >>> d['anotherthing'] = 'bar'
    >>> print(d['something'])
    foo
    >>> print(d.anotherthing)
    bar
    >>> def babar():
    >>>     print('babar !')
    >>> d.babar = babar
    >>> d.babar()
    babar !
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

# Aliases for AttributeDict
AD = AttributeDict
Dataset = AttributeDict


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


def make_dataset(X, y, batchsize):
    """
    Build a dictionnay dataset from the given arrays
    60% : train
    15% : validation
    25% : test

    Params
    ------
        X : the data (numpy array)
        y : the labels (numpy array)
        batchsize : the batchsize (int)
    Return
    ------
        dataset : An AttributeDict with
                    X_train: training data (60%),
                    y_train: training labels,
                    X_val: validation data (15%),
                    y_val: validation labels,
                    X_test: testing data (25%),
                    y_test: testing labels,
                    batchsize: batchsize,
    """
    n_samples = X.shape[0]
    n_train = int(0.6*n_samples)
    n_val = int(0.15*n_samples)+n_train

    X_train, X_val, X_test = X[0:n_train], X[n_train:n_val], X[n_val:]
    y_train, y_val, y_test = y[0:n_train], y[n_train:n_val], y[n_val:]
    
    # Note to self : 
    #   You should still use the incomming datasets as if they were simple dictionaries
    #   In order to prevent breaking the old working codes
    dataset = Dataset({
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    })
    return dataset



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
    # Note to self : 
    #   You should still use the incomming datasets as if they were simple dictionaries
    #   In order to prevent breaking the old working codes
    domain_data = Dataset({
            'X_train': [data['X_train'] for data in datasets],
            'X_val': [data['X_val'] for data in datasets],
            'X_test': [data['X_test'] for data in datasets],
            'y_train': None,
            'y_val': None,
            'y_test': None,
            'batchsize': datasets[0]['batchsize'],
            })

    return domain_data


def make_corrector_dataset(source_data, target_data):    
    """
    Make a corrector datasets out of the given datasets.

    Params
    ------
        datasets: a list of datasets (dicts with the separated data)

    Return
    ------
        domain_data: dict with the separated data
    """
    # Note to self : 
    #   You should still use the incomming datasets as if they were simple dictionaries
    #   In order to prevent breaking the old working codes
    corrector_data = Dataset(target_data)
    corrector_data.update({
        'y_train': source_data['X_train'],
        'y_val': source_data['X_val'],
        'y_test': source_data['X_test'],
        'labels': source_data['y_train'],
        'batchsize': source_data['batchsize'],
        })
    return corrector_data

