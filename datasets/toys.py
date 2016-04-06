#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from utils import shuffle_array, make_domain_dataset
from sklearn.datasets import make_moons

# ============================================================================
#                   Moon & Moon rotated
# ============================================================================


def rotate_data(X, angle=35.):
    """Apply a rotation on a 2D dataset.
    """
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    X_r = np.empty_like(X)
    X_r[:] = X[:].dot(rotMatrix)
    return X_r


def load_moon(noise=0.05, angle=35., batchsize=32):
    """
    Load the Moon / Moon-rotated problem

    Params
    ------
        noise: (default=0.05) the noise of the moon data generator
        angle: (default=35.0) the angle (in degree) of the rotated Moons
        shape: (default=(-1, 28, 28, 3)) the output shape of the data.
            Should be (-1, 3, 28, 28) to be used by convolution layers.
    
    Return
    ------
        source_data: dict with the separated data
        target_data: dict with the separated data
        domain_data: dict with the separated data

    """
    X, y = make_moons(n_samples=500, shuffle=True, noise=noise, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X_r = rotate_data(X, angle=angle)

    X_S, y_S, X_T, y_T = shuffle_array(X, y, X_r, y)
    # X_T, y_T = shuffle_array(X_r, y)

    X_train, X_val, X_test = X_S[0:300], X_S[300:400], X_S[400:]
    y_train, y_val, y_test = y_S[0:300], y_S[300:400], y_S[400:]
    
    X_t_train, X_t_val, X_t_test = X_T[0:300], X_T[300:400], X_T[400:]
    y_t_train, y_t_val, y_t_test = y_T[0:300], y_T[300:400], y_T[400:]
    
    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize':batchsize,
                    }

    target_data = {
                    'X_train': X_t_train,
                    'y_train': y_t_train,
                    'X_val': X_t_val,
                    'y_val': y_t_val,
                    'X_test': X_t_test,
                    'y_test': y_t_test,
                    'batchsize':batchsize,
                    }

    domain_data = make_domain_dataset([source_data, target_data])
    return source_data, target_data, domain_data


def load_clouds(n_sample=500 ,n_classes=2, batchsize=50):
    """
    Dataset made from normal distributions localised at root of unity solution.

    Params
    ------
        n_sample: (default=500) the number of trainning sample in each class
            (validation and test have n_sample/2 samples)
        n_classes: (default=2) the number of classes
        batchsize: the batchsize
    
    Return
    ------
        source_data: dict with the separated data
    """
    pos = [np.exp(2j*np.pi*i/n_classes) for i in range(n_classes)]

    X_train = np.empty((n_sample*n_classes, 2))
    X_train[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_sample) for p in pos])
    X_train[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_sample) for p in pos])
    y_train = np.hstack([np.ones(n_sample)*i for i in range(n_classes)])

    X_val = np.empty((n_sample*n_classes, 2))
    X_val[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_sample) for p in pos])
    X_val[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_sample) for p in pos])
    y_val = np.hstack([np.ones(n_sample)*i for i in range(n_classes)])

    X_test = np.empty((n_sample*n_classes, 2))
    X_test[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_sample) for p in pos])
    X_test[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_sample) for p in pos])
    y_test = np.hstack([np.ones(n_sample)*i for i in range(n_classes)])

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
                    'batchsize':batchsize,
                    }
    return source_data

if __name__ == '__main__':
    load_moon()