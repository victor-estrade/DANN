#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import gzip
import sys
import os
from glob import glob
import numpy as np
import cPickle as pickle
# import pandas as pd
import matplotlib.pyplot as plt

# import nltk
# import sklearn
from scipy import misc
from sklearn.datasets import make_moons as mkmoon


np.random.seed(12345)


def shuffle(X, y,):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


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
    X, y = mkmoon(n_samples=500, shuffle=True, noise=noise, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X_r = rotate_data(X, angle=angle)

    X_S, y_S = shuffle(X, y)
    X_T, y_T = shuffle(X_r, y)

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

    domain_data = {
                    'X_train': np.vstack([X_train, X_t_train]),
                    'y_train': np.hstack([np.zeros_like(y_train, dtype=np.int32), 
                               np.ones_like(y_t_train, dtype=np.int32)]),
                    'X_val': np.vstack([X_val, X_t_val]),
                    'y_val': np.hstack([np.zeros_like(y_val, dtype=np.int32), 
                               np.ones_like(y_t_val, dtype=np.int32)]),
                    'X_test': np.vstack([X_test, X_t_test]),
                    'y_test': np.hstack([np.zeros_like(y_test, dtype=np.int32), 
                               np.ones_like(y_t_test, dtype=np.int32)]),
                    'batchsize':batchsize*2,
                    }
    return source_data, target_data, domain_data


# ============================================================================
#                   MNIST & MNIST-M
# ============================================================================
# Get every images files from the BSR-BSDS500 training dataset
bsr = 'data/BSR/BSDS500/data/images/train/*.jpg'
bsr = glob(bsr)


def to_rgb(im):
    """
    Add a new axis/dimension of size 3 by repeating the content of the array.

    Params
    ------
        im: (numpy array [N,M,...,Z]) the input array/image to be extended

    Return
    ------
        extended_im: (numpy array [N,M,...,Z,3]) the extended array/image with
            a new dimension. 
    """
    ret = np.empty(im.shape+(3,))
    ret[..., 2] = ret[..., 1] = ret[..., 0] = im
    return ret


def patch():
    """
    Extract a 28x28 pixels patch from a random location of a random image 
    taken in the BSDS500 training dataset.
    Return
    ------
        extract: (numpy array [28,28,3]) a piece of a random image.
    """
    c = np.random.randint(len(bsr))
    img = misc.imread(bsr[c])/255
    i = np.random.randint(img.shape[0]-28)
    j = np.random.randint(img.shape[1]-28)
    extract = np.asarray(img[i:i+28, j:j+28, :])
    return extract


def blend(img1, img2):
    """
    Merge two images:
        img_final = |img_1 - img_2|

    Warning : Do not input uint8 arrays, the computation is overflow sensitive.

    Params
    ------
        img1: (numpy array like) should have the same shape as img2
        img2: (numpy array like) should have the same shape as img1
    Return
    ------
        img_final: 
    """
    return np.abs(img1-img2)


def mnist_blend(data):
    """
    Transform the given MNIST data to RGB image and computes MNIST-M
    """
    data = to_rgb(data.reshape(-1, 28, 28))
    new = np.empty_like(data)
    new[:, ...] = blend(data[:], patch())
    data = data.reshape(data.shape[0], -1)
    new = new.reshape(new.shape[0], -1)
    return data, new


def load_mnist():
    """
    Load the raw MNIST dataset

    Return
    ------
        train, valid, test: the datasets, couples of numpy arrays (X, y). 
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_S, valid_S, test_S = pickle.load(f)
    f.close()
    return train_S, valid_S, test_S


def load_mnistM(shape=(-1, 28, 28, 3)):
    """
    Load the MNIST / MNIST-M problem

    Params
    ------
        shape: (default=(-1, 28, 28, 3)) the output shape of the data.
            Should be (-1, 3, 28, 28) to be used by convolution layers.
    Return
    ------
        source_data: dict with the separated data
        target_data: dict with the separated data
        domain_data: dict with the separated data

    """
    source = load_mnist() # Load the raw MNIST data
    train_S, valid_S, test_S = source
    # Blend the MNIST image to build the MNIST-M dataset
    data = tuple((mnist_blend(X) + (y,) for X, y in source))
    target = tuple(((d[1].reshape(shape), d[2]) for d in data))
    source = tuple(((d[0].reshape(shape), d[2]) for d in data))

    train_S, val_S, test_S = source
    train_T, val_T, test_T = target

    X_train, y_train = train_S
    X_t_train, y_t_train = train_T

    X_val, y_val = val_S
    X_t_val, y_t_val = val_T

    X_test, y_test = test_S
    X_t_test, y_t_test = test_T

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

    domain_data = {
                    'X_train': np.vstack([X_train, X_t_train]),
                    'y_train': np.hstack([np.zeros_like(y_train, dtype=np.int32), 
                               np.ones_like(y_t_train, dtype=np.int32)]),
                    'X_val': np.vstack([X_val, X_t_val]),
                    'y_val': np.hstack([np.zeros_like(y_val, dtype=np.int32), 
                               np.ones_like(y_t_val, dtype=np.int32)]),
                    'X_test': np.vstack([X_test, X_t_test]),
                    'y_test': np.hstack([np.zeros_like(y_test, dtype=np.int32), 
                               np.ones_like(y_t_test, dtype=np.int32)]),
                    'batchsize':batchsize*2,
                    }

    return source_data, target_data, domain_data


if __name__ == '__main__':
    print('I am at your service, master.')
    source, target = load_mnistM()
    train_S, val_S, test_S = source
    train_T, val_T, test_T = target
    X_S, y_S = train_S
    X_T, y_T = train_T

    i = np.random.randint(X_S.shape[0])

    plt.imshow(X_S[i])
    plt.title('AVANT-label='+str(y_S[i]))
    plt.show()
    plt.imshow(X_T[i])
    plt.title('APRES-label='+str(y_T[i]))
    plt.show()

    
    # SAVE DATA
    # if not os.path.isdir('data/train/'):
    #     os.mkdir('data/train/')

    # with open('data/train/minist.pkl', 'wb') as f:
    #     pickle.dump({'X': X, 'y':y}, f)

    # np.save('data/train/minist', X)
    # np.save('data/train/minist-m', new_X)
