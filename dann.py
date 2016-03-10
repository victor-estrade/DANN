#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import time
import gzip

import theano
import lasagne

import cPickle as pickle
import numpy as np
import theano.tensor as T
# import pandas as pd
import matplotlib.pyplot as plt

from datasets import mnist_blend
from logs import log_fname, new_logger
"""
http://stackoverflow.com/questions/33879736/can-i-selectively-invert-theano-gradients-during-backpropagation
"""


class ReverseGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


class ReverseGradientLayer(lasagne.layers.Layer):
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.op = ReverseGradient(hp_lambda)

    def get_output_for(self, input, **kwargs):
        return self.op(input)

# BUILD FACTORY :
factory_dict = {}

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    cv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    cv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(feature, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return network
# Add this builder to the factory
factory_dict['cnn'] = build_cnn

def build_dann(input_var=None, hp_lambda=0.5):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    cv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 48 5x5 kernels, and another 2x2 pooling:
    cv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    label_predictor = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(feature, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    dense_domain = lasagne.layers.DenseLayer(
            ReverseGradientLayer(pool2, hp_lambda=hp_lambda),
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    domain_predictor = lasagne.layers.DenseLayer(
            dense_domain,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor
# Add this builder to the factory
factory_dict['dann'] = build_dann


def build_small_dann(input_var=None, hp_lambda=0.5):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    # cv1 = lasagne.layers.Conv2DLayer(
    #         input_layer, num_filters=32, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         # W=lasagne.init.GlorotUniform(),
    #         )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 48 5x5 kernels, and another 2x2 pooling:
    # cv2 = lasagne.layers.Conv2DLayer(
    #         pool1, num_filters=48, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         # W=lasagne.init.GlorotUniform(),
    #         )
    # pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            input_layer,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    label_predictor = lasagne.layers.DenseLayer(
            feature,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    domain_hidden = lasagne.layers.DenseLayer(
            ReverseGradientLayer(feature, hp_lambda=hp_lambda),
            num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    domain_predictor = lasagne.layers.DenseLayer(
            ReverseGradientLayer(feature, hp_lambda=hp_lambda),
            # domain_hidden,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor
# Add this builder to the factory
factory_dict['small'] = build_small_dann


def build_factory(name='dann', **kwargs):
    """
    Helper function to build pre-set theano neural networks graph

    Params
    ------
        name: the name of the neural network / builder function. (string)
        **kwargs: other key word arguments can be passed to the builder 
            function.
    Return
    ------
        the datasets
    """
    if name in factory_dict:
        return factory_dict[name](**kwargs)
    else:
        raise NotImplementedError(
            "This neural network ({}) is not implemented yet. "
            "Please check the model name spelling.".format(name))


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Helper function interating over the given inputs

    Params
    ------
        inputs: the data (numpy array)
        targets: the target values (numpy array)
        batchsize: the batch size (int)
        shuffle (default=False):
    
    Return
    ------
        (input_slice, target_slice) as a generator
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def test(predict_fun, X, y, logger=None, batchsize=500):
    """
    Test the model using the given predict function on minibacth.
    Return the statistics.

    Params
    ------
        predict_fun: the predict function. Should take minibatch from X and y
            and return a loss value and an accuracy value :
            >>> loss, accuracy =  predict_fun(X, y)
        X: the input data
        y: the target value
        logger (default=None): used to output some information
        batchsize (default=500): the size on the minibatches

    Return
    ------
        stats: a dictionnary with 'loss' and 'acc'
    """
    # After training, we compute and print the test error:
    stats = {}
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X, y, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = predict_fun(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_err /= test_batches
    test_acc = test_acc / test_batches * 100
    if logger:
        logger.info("  {:30}: {:.6f}".format('source test loss',
            test_err))
        logger.info("  {:30}: {:.2f} %".format('source test accuracy',
            test_acc))

    # And saving them:
    stats['loss'] = test_err
    stats['acc'] = test_acc
    return stats


if __name__ == '__main__':
    g = build_factory('cnn')
    print(type(g))
    g = build_factory('small')
    print(type(g))
    print(type(g) is tuple)
    g = build_factory('sm')
