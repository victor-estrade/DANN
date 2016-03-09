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


def to_rgb(im):
    ret = np.empty(im.shape+(3,))
    ret[..., 2] = ret[..., 1] = ret[..., 0] = im
    return ret


def patch(random_state=12345):
    rng = np.random.RandomState(random_state)
    c = rng.randint(len(bsr))
    img = misc.imread(bsr[c])/255
    i = rng.randint(img.shape[0]-28)
    j = rng.randint(img.shape[1]-28)
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

if __name__ == '__main__':
    print('I am at your service, master.')
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_S, valid_S, test_S = pickle.load(f)
    f.close()

    X, y = train_S
    
    X, new_X = mnist_blend(X)
    
    i = np.random.randint(X.shape[0])

    plt.imshow(X[i].reshape(28, 28, 3))
    plt.title('AVANT-'+str(y[i]))
    plt.show()
    plt.imshow(new_X[i].reshape(28, 28, 3))
    plt.title('APRES-'+str(y[i]))
    plt.show()

    X = X.reshape(X.shape[0], -1)
    new_X = new_X.reshape(new_X.shape[0], -1)

    # SAVE DATA
    # if not os.path.isdir('data/train/'):
    #     os.mkdir('data/train/')

    # with open('data/train/minist.pkl', 'wb') as f:
    #     pickle.dump({'X': X, 'y':y}, f)

    # np.save('data/train/minist', X)
    # np.save('data/train/minist-m', new_X)
