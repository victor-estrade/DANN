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

__author__ = 'Estrade Victor'

bsr = 'data/BSR/BSDS500/data/images/train/*.jpg'
bsr = glob(bsr)
np.random.seed(12345)


def to_rgb(im):
    ret = np.empty(im.shape+(3,))
    ret[..., 2] = ret[..., 1] = ret[..., 0] = im
    return ret


def patch():
    c = np.random.randint(len(bsr))
    img = misc.imread(bsr[c])/255
    i = np.random.randint(img.shape[0]-28)
    j = np.random.randint(img.shape[1]-28)
    extract = np.asarray(img[i:i+28, j:j+28, :])
    return extract


def blend(img1, img2):
    return np.abs(img1-img2)


def mnist_blend(data):
    data = to_rgb(data.reshape(-1, 28, 28))
    new = np.empty_like(data)
    new[:, ...] = blend(data[:], patch())
    data = data.reshape(data.shape[0], -1)
    new = new.reshape(new.shape[0], -1)
    return data, new


def load_mnist():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_S, valid_S, test_S = pickle.load(f)
    f.close()
    return train_S, valid_S, test_S


def load_mnistM(shape=(-1, 28, 28, 3)):
    source = load_mnist()
    train_S, valid_S, test_S = source
    data = tuple((mnist_blend(X) + (y,) for X, y in source))
    target = tuple(((d[1].reshape(shape), d[2]) for d in data))
    source = tuple(((d[0].reshape(shape), d[2]) for d in data))
    return source, target


if __name__ == '__main__':
    print('I am at your service, master.')
    source, target = load_mnistM()
    train_S, val_S, test_S = source
    train_T, val_T, test_T = target
    X_S, y_S = train_S
    X_T, y_T = train_T

    i = np.random.randint(X_S.shape[0])

    plt.imshow(X_S[i])
    plt.title('AVANT-'+str(y_S[i]))
    plt.show()
    plt.imshow(X_T[i])
    plt.title('APRES-'+str(y_T[i]))
    plt.show()

    
    # SAVE DATA
    # if not os.path.isdir('data/train/'):
    #     os.mkdir('data/train/')

    # with open('data/train/minist.pkl', 'wb') as f:
    #     pickle.dump({'X': X, 'y':y}, f)

    # np.save('data/train/minist', X)
    # np.save('data/train/minist-m', new_X)
