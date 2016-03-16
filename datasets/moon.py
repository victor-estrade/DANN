#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from utils import shuffle_array, domain_X_y
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

    X_S, y_S = shuffle_array(X, y)
    X_T, y_T = shuffle_array(X_r, y)

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

    X_train , y_train = domain_X_y([X_train, X_t_train])
    X_val , y_val = domain_X_y([X_val, X_t_val])
    X_test , y_test = domain_X_y([X_test, X_t_test])
    domain_data = {
                    'X_train': X_train,
                    'y_train': y_train, 
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize':batchsize*2,
                    }

    return source_data, target_data, domain_data

if __name__ == '__main__':
    load_moon()