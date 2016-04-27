#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from utils import shuffle_array
from sklearn.datasets import make_moons

# ============================================================================
#                   Moon & Moon rotated
# ============================================================================


def load_moons(noise=0.05, n_samples=500, batchsize=32):
    """
    Load the Moon dataset using sklearn.datasets.make_moons() function.

    Params
    ------
        noise: (default=0.05) the noise of the moon data generator
        n_samples: (default=500) the total number of points generated
        batchsize: (default=32) the dataset batchsize
    
    Return
    ------
        source_data: dict with the separated data

    """
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    X, y = shuffle_array(X, y)  # Usefull ?

    n_train = int(0.4*n_samples)
    n_val = int(0.3*n_samples)+n_train

    X_train, X_val, X_test = X[0:n_train], X[n_train:n_val], X[n_val:]
    y_train, y_val, y_test = y[0:n_train], y[n_train:n_val], y[n_val:]
    
    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data


def load_clouds(n_samples=50 ,n_classes=2, batchsize=5):
    """
    Dataset made from normal distributions localised at root of unity solution.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 training points 150 validation points 
            and 150 test points
        n_classes: (default=2) the number of classes
        batchsize: the batchsize of the dataset
    
    Return
    ------
        source_data: dict with the separated data
    """
    # pos is the 2D positions as complex exponential numbers, root of unity solutions
    pos = [np.exp(2j*np.pi*i/n_classes) for i in range(n_classes)]

    X_train = np.empty((n_samples*n_classes, 2))
    X_train[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_train[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_train = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_val = np.empty((n_samples*n_classes, 2))
    X_val[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_val[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_val = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_test = np.empty((n_samples*n_classes, 2))
    X_test[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_test[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_test = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_train, y_train = shuffle_array(X_train, y_train)
    X_val, y_val = shuffle_array(X_val, y_val)
    X_test, y_test = shuffle_array(X_test, y_test)

    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data


if __name__ == '__main__':
    load_moon()