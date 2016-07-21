# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib.pyplot as plt

np.random.seed(12345)

data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, 'data')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def _load_mnist():
    """
    Load the raw MNIST dataset

    Return
    ------
        train, test: the datasets, couples of numpy arrays (X, y). 
    """
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(data_dir, filename))

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(os.path.join(data_dir, filename)):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(os.path.join(data_dir, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(os.path.join(data_dir, filename)):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(os.path.join(data_dir, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    """
    Load the MNIST dataset

    Return
    ------
        X: The data (numpy array shape [70 000, 1, 28, 28])
        y: The labels (numpy array shape [70 000])
    """
    train_S, test_S = _load_mnist() # Load the raw MNIST data

    X_train, y_train = train_S
    X_test, y_test = test_S
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y
