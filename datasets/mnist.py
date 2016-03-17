#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import gzip
import sys
import os

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from utils import domain_X_y
from glob import glob
from scipy import misc

np.random.seed(12345)

# ============================================================================
#                   MNIST
# ============================================================================
def load_mnist():
    """
    Load the raw MNIST dataset

    Return
    ------
        train, valid, test: the datasets, couples of numpy arrays (X, y). 
    """
    f = gzip.open(os.path.join(data_dir, 'mnist.pkl.gz'), 'rb')
    train_S, valid_S, test_S = pickle.load(f)
    f.close()
    return train_S, valid_S, test_S

# ============================================================================
#                   MNIST-M
# ============================================================================
data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, 'data')

# Get every images files from the BSR-BSDS500 training dataset
bsr = os.path.join(data_dir, 'BSR/BSDS500/data/images/train/*.jpg')
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

    Warning : Do not input uint8 arrays, the computation is overflow sensitive

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


def load_mnistM(roll=True, batchsize=500):
    """
    Load the MNIST / MNIST-M problem

    Params
    ------
        roll: (default=True) roll the output shape of the data. 
            Default shape = (-1, 28, 28, 3) but if roll is True it 
            will be (-1, 3, 28, 28) to be used by convolution layers.
    Return
    ------
        source_data: dict with the separated data
        target_data: dict with the separated data
        domain_data: dict with the separated data

    """
    source = load_mnist() # Load the raw MNIST data
    # Blend the MNIST image to build the MNIST-M dataset
    data = tuple((mnist_blend(X) + (y,) for X, y in source))
    target = tuple(((np.rollaxis(d[1], 3, 1), d[2]) for d in data))
    source = tuple(((np.rollaxis(d[0], 3, 1), d[2]) for d in data))

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


# ============================================================================
#                   MNIST-invert black & white
# ============================================================================

def load_mnist_invert(roll=True, batchsize=500):
    """
    Load the MNIST / 1-MNIST problem

    Params
    ------
        TODO : Roll ou shape ?
        batchsize: (default=500) the batch size.

    Return
    ------
        source_data: dict with the separated data
        target_data: dict with the separated data
        domain_data: dict with the separated data

    """
    source = load_mnist() # Load the raw MNIST data
    train_S, val_S, test_S = source
    
    X_train, y_train = train_S
    X_val, y_val = val_S
    X_test, y_test = test_S

    X_train = X_train.reshape(-1, 28, 28)
    X_val = X_val.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    X_t_val, y_t_val = (1-X_val), y_val
    X_t_train, y_t_train = (1-X_train), y_train
    X_t_test, y_t_test = (1-X_test), y_test
    
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


# ============================================================================
#                   MNIST-Mirror
# ============================================================================

def load_mnist_mirror(roll=True, batchsize=500):
    """
    Load the MNIST / 1-MNIST problem

    Params
    ------
        shape: (default=(-1, 1, 28, 28)) the output shape of the data.
        batchsize: (default=500) the batch size.

    Return
    ------
        source_data: dict with the separated data
        target_data: dict with the separated data
        domain_data: dict with the separated data

    """
    source = load_mnist() # Load the raw MNIST data
    train_S, val_S, test_S = source
    
    X_train, y_train = train_S
    X_val, y_val = val_S
    X_test, y_test = test_S
    
    X_train = X_train.reshape(-1, 28, 28)
    X_val = X_val.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    X_t_train, y_t_train = np.fliplr(X_train), y_train
    X_t_val, y_t_val = np.fliplr(X_val), y_val
    X_t_test, y_t_test = np.fliplr(X_test), y_test
    
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
    print('I am at your service, master.')
    # source, target, domain = load_mnistM()
    # source, target, domain = load_mnist_invert(roll=False)
    source, target, domain = load_mnist_mirror()
    X_S, y_S = source['X_train'], source['y_train']
    X_T, y_T = target['X_train'], target['y_train']
    np.random.seed(None)
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
