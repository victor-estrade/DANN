#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from utils import domain_X_y, make_domain_dataset
from glob import glob
from scipy import misc

np.random.seed(12345)

data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, 'data')
if not os.path.isdir(data_dir):
	os.mkdir(data_dir)
# Get every images files from the BSR-BSDS500 training dataset
bsr = os.path.join(data_dir, 'BSR/BSDS500/data/images/train/*.jpg')
bsr = glob(bsr)
# Mnist-M path
mnistM_path = os.path.join(data_dir, 'mnistM.pkl.gz')

# ============================================================================
#                   MNIST
# ============================================================================

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_mnist():
    """
    Load the raw MNIST dataset

    Return
    ------
        train, valid, test: the datasets, couples of numpy arrays (X, y). 
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

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    f = gzip.open(os.path.join(data_dir, 'mnist.pkl.gz'), 'rb')
    train_S, valid_S, test_S = pickle.load(f)
    f.close()
    return train_S, valid_S, test_S


def load_mnist_src(batchsize=500, shape=(-1, 28, 28)):
    """
    Load the MNIST / 1-MNIST problem

    Params
    ------
        batchsize: (default=500) the batch size.

    Return
    ------
        source_data: dict with the separated data

    """
    train_S, val_S, test_S = load_mnist() # Load the raw MNIST data

    X_train, y_train = train_S
    X_val, y_val = val_S
    X_test, y_test = test_S
    
    X_train = X_train.reshape(shape)
    X_val = X_val.reshape(shape)
    X_test = X_test.reshape(shape)
 
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


# ============================================================================
#                   MNIST-M
# ============================================================================


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
    for i in range(new.shape[0]):
        if i % 1000 == 0:
            print(i, '/', new.shape[0])
        new[i, ...] = blend(data[i], patch())
    return new


def build_mnistM():
    """
    Build the Mnist-M dataset
    """
    print('Building the MNIST-M dataset')
    source = load_mnist() # Load the raw MNIST data
    # Blend the MNIST image to build the MNIST-M dataset
    target = tuple(((np.rollaxis(mnist_blend(X), 3, 1), y) for X, y in source))
    train_T, val_T, test_T = target

    X_t_train, y_t_train = train_T
    X_t_val, y_t_val = val_T
    X_t_test, y_t_test = test_T
    target_data = {
                    'X_train': X_t_train,
                    'y_train': y_t_train,
                    'X_val': X_t_val,
                    'y_val': y_t_val,
                    'X_test': X_t_test,
                    'y_test': y_t_test,
                    'batchsize':500,
                    }
    print('Saving MNIST-M dataset to', mnistM_path)
    f = gzip.open(mnistM_path,'wb')
    pickle.dump(target_data, f)
    f.close()
    return target_data


def load_mnistM():
    if os.path.isfile(mnistM_path):
        f = gzip.open(mnistM_path, 'rb')
        target = pickle.load(f)
        f.close()
        return target
    else:
        target = build_mnistM()
        return target


def load_mnist_M(roll=True, batchsize=500):
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
    raise NotImplementedError('Big Bug found. Fix in progress.')
    source = load_mnist() # Load the raw MNIST data
    
    source = tuple(((np.rollaxis(to_rgb(X.reshape(-1, 28, 28)), 3, 1), y)
                    for X, y in source))

    train_S, val_S, test_S = source
    X_train, y_train = train_S
    X_val, y_val = val_S
    X_test, y_test = test_S

    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize':batchsize,
                    }
    target_data = load_mnistM()
    target_data['batchsize'] = batchsize
    domain_data = make_domain_dataset([source_data, target_data])
    return source_data, target_data, domain_data


# ============================================================================
#                   MNIST-invert black & white
# ============================================================================

def load_mnist_invert(roll=True, batchsize=500, shape=(-1, 28, 28)):
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

    X_train = X_train.reshape(shape)
    X_val = X_val.reshape(shape)
    X_test = X_test.reshape(shape)

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

    domain_data = make_domain_dataset([source_data, target_data])
    
    return source_data, target_data, domain_data


# ============================================================================
#                   MNIST-Mirror
# ============================================================================

def load_mnist_mirror(roll=True, batchsize=500, shape=(-1, 28, 28)):
    """
    Load the MNIST / 1-MNIST problem

    Params
    ------
        TODO roll vs shape problem 
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
    
    X_train = X_train.reshape(shape)
    X_val = X_val.reshape(shape)
    X_test = X_test.reshape(shape)

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

    domain_data = make_domain_dataset([source_data, target_data])
    
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
