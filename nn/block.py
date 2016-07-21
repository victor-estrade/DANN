#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import lasagne

from rgl import ReverseGradientLayer

__ACTIVATIONS = {
    'sigmoid': lasagne.nonlinearities.sigmoid,
    'identity': lasagne.nonlinearities.identity,
    'leaky_rectify': lasagne.nonlinearities.leaky_rectify,
    'linear': lasagne.nonlinearities.linear,
    'rectify': lasagne.nonlinearities.rectify,
    'relu': lasagne.nonlinearities.rectify,
    'sigmoid': lasagne.nonlinearities.sigmoid,
    'softmax': lasagne.nonlinearities.softmax,
    'tanh': lasagne.nonlinearities.tanh,
}


def dense(input_layer, n_neurons, activation=lasagne.nonlinearities.sigmoid):
    """
    Build a multiple dense layer block :

    input_layer-->dense_0-->...-->dense_n

    Params
    ------
        input_layer: the layer on which will be append the multiple dense block.
        n_neurons: the list containing the number of neuron for each dense layers.
        activation: (default=sigmoid) the activation
            ('sigmoid', 'identity', 'leaky_rectify', 'linear', 'rectify', 
            'relu', 'sigmoid', 'softmax', 'tanh') or a lasagne.nonlinearities
            or a custom activation
    Return
    ------
        dense_n: the last dense layer
    """
    try:
        activation = __ACTIVATIONS[activation]
    except KeyError, e:
        pass
    layer = input_layer
    for n in n_neurons:
        layer = lasagne.layers.DenseLayer(
                    layer,
                    num_units=n,
                    nonlinearity=activation,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
    return layer


def multi_proba(feature_layer, n_classes_list):
    """
    Build a multiple probability predictor :

                   +-->softmax--+
                   |            |
                   +    ....    +
                   |            |
    feature_layer -+-->softmax--+-->concat(axis=1) 
                   |            |
                   +    ....    +
                   |            |
                   +-->softmax--+

    Params
    ------
        feature_layer: the layer on which will be append the multiple softmax + concat block.
        n_class_list: the list containing the number of neuron for each softmax part.
    Return
    ------
        concat_layer: the concat layer at the end of the stucture
    """
    multi = [lasagne.layers.DenseLayer(
                feature_layer,
                num_units=n_classes,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                )
            for n_classes in n_classes_list]
    concat_layer = lasagne.layers.ConcatLayer(multi, axis=1)
    return concat_layer
