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

from rgl import ReverseGradientLayer
from datasets import mnist_blend
from logs import log_fname, new_logger

# BUILD FACTORY :
factory_dict = {}


def build_cnn(input_var=None, shape=(None, 3, 28, 28), **kwargs):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=shape,
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


def build_dann(input_var=None, hp_lambda=0.5, shape=(None, 3, 28, 28)):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    cv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform('relu'),
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
            W=lasagne.init.GlorotUniform('relu'),
            )
    pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    feature = pool2
    # Reversal gradient layer
    RGL = ReverseGradientLayer(feature, hp_lambda=hp_lambda)
    
    # Label Pedictor
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    label_dense = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(feature, p=.5),
            num_units=100,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )
    label_predictor = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(label_dense, p=.5),
            # lasagne.layers.dropout(feature, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    # dense_domain = lasagne.layers.DenseLayer(
    #         RGL,
    #         num_units=100,
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform(),
    #         )

    domain_predictor = lasagne.layers.DenseLayer(
            RGL,
            # dense_domain,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor
# Add this builder to the factory
factory_dict['dann'] = build_dann


def build_small_dann(input_var=None, hp_lambda=0.5, shape=(None, 3, 28, 28)):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)

    # A fully-connected layer of 256 units
    feature = lasagne.layers.DenseLayer(
            input_layer,
            num_units=6,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.Uniform(range=0.1, std=None, mean=0.0),
            )
    feature = lasagne.layers.DenseLayer(
            feature,
            num_units=50,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.Uniform(range=0.1, std=None, mean=0.0),
            )

    # Reversal gradient layer
    RGL = ReverseGradientLayer(feature, hp_lambda=hp_lambda)
    
    # Label classifier
    # label_hidden = lasagne.layers.DenseLayer(
    #         feature,
    #         num_units=50,
    #         nonlinearity=lasagne.nonlinearities.sigmoid,
    #         W=lasagne.init.GlorotUniform(),
    #         )
    label_predictor = lasagne.layers.DenseLayer(
            # input_layer,
            feature,
            # label_hidden,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    # domain_hidden = lasagne.layers.DenseLayer(
    #         RGL,
    #         num_units=50,
    #         nonlinearity=lasagne.nonlinearities.sigmoid,
    #         W=lasagne.init.GlorotUniform(),
    #         )
    domain_predictor = lasagne.layers.DenseLayer(
            RGL,
            # domain_hidden,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
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


if __name__ == '__main__':
    g = build_factory('cnn')
    print(type(g))
    g = build_factory('small')
    print(type(g))
    print(type(g) is tuple)
    g = build_factory('sm')
