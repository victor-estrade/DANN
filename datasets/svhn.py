#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import gzip
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

np.random.seed(12345)

data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, 'data')

def load_svhn_src(roll=True, batchsize=600):
    """
    TODO : read again and rework it !
    """
    data = io.loadmat(os.path.join(data_dir,'train_32x32.mat'))
    X = data['X']
    y = data['y']
    X = np.rollaxis(X, 3)
    if roll:
        X = np.rollaxis(X, 3, 1)

    s1 = 50000
    s2 = 60000
    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]

    data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'batchsize': batchsize,
            }
    return data

if __name__ == '__main__':
    load_svhn_src()
