#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import time
import abc

import theano
import lasagne

import numpy as np
import theano.tensor as T

from rgl import ReverseGradientLayer
from logs import log_fname, new_logger
from utils import iterate_minibatches


class AbstractBlock(lasagne.layers.Layer):
    """
    The base class for Blocks.

    TODO: Make the Blocks act likes a real Layer.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, incoming, **kwargs):
        incoming = self.get_input_layer(incoming)
        super(AbstractBlock, self).__init__(incoming, **kwargs)
        self.input_layer = incoming
        self.output_layer = None

    @abc.abstractmethod
    def clone(self, input_layer) :
        """
        Create a clone of the block sharing its weights 
        but taking a different input.
        """
        return

    def get_input_layer(self, input_layer):
        if isinstance(input_layer, AbstractBlock):
            return input_layer.output_layer
        elif isinstance(input_layer, lasagne.layers.Layer):
            return input_layer
        else:
            raise ValueError("Input_layer should be a lasagne.layer.Layer "
                             "or a Block. {} found".format(type(input_layer)))


class Dense(AbstractBlock):
    """
    A multiple dense layers.
        
    Params
    ------
        input_layer: the inputs
        arch: a list of integer giving the number of nerones in the layers.
        nonlinearity: (default=tanh) the nonlinearity activation on every layers.

    """
    def __init__(self, incoming, arch, nonlinearity=lasagne.nonlinearities.tanh, **kwargs):
        incoming = self.get_input_layer(incoming)
        super(Dense, self).__init__(incoming, **kwargs)
        # Save everything to be able to clone
        self.arch = arch
        self.nonlinearity = nonlinearity
        self.kwargs = kwargs

        self.layers = []
        l_tmp = self.input_layer
        for nb_units in arch:
            l_tmp = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_units,
                    nonlinearity=nonlinearity,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
            self.layers.append(l_tmp)            
        self.output_layer = l_tmp

    def clone(self, input_layer):
        """
        Create a clone of the block sharing its weights 
        but taking a different input.
        """
        clone = Dense(input_layer, self.arch, self.nonlinearity, **self.kwargs)
        for layer_A, layer_clone in zip(self.layers, clone.layers):
            layer_clone.W = layer_A.W
            layer_clone.b = layer_A.b
        clone.output_layer = clone.layers[-1]

        return clone


class Classifier(AbstractBlock):
    """
    A multiple dense layers ending by a classifier.
        
    Params
    ------
        input_layer: the inputs
        nb_label : the number of label to be predicted. 
            (number of neurone in the last softmax layer)
        arch: (default=[]) a list of integer giving the number of nerones in the layers.

    """
    def __init__(self, incoming, nb_label, arch=[], 
                nonlinearity=lasagne.nonlinearities.tanh, **kwargs):
        super(Classifier, self).__init__(incoming, **kwargs)
        # Save everything to be able to clone
        self.nb_label = nb_label
        self.arch = arch
        self.nonlinearity = nonlinearity
        self.kwargs = kwargs
        
        self.layers = []
        l_tmp = self.input_layer
        for nb_units in arch:
            l_tmp = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_units,
                    nonlinearity=nonlinearity,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
            self.layers.append(l_tmp)
        self.output_layer = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_label,
                    nonlinearity=lasagne.nonlinearities.softmax,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )

    def clone(self, input_layer):
        """
        Create a clone of the block sharing its weights 
        but taking a different input.
        """
        clone = Classifier(input_layer, self.nb_label, self.arch, **self.kwargs)
        for layer_A, layer_clone in zip(self.layers, clone.layers):
            layer_clone.W = layer_A.W
            layer_clone.b = layer_A.b
        clone.output_layer.W = self.output_layer.W
        clone.output_layer.b = self.output_layer.b
        
        return clone


def adversarial(layers, hp_lambda=1, lr=1, mom=.9):
    """
    Stochastic Gradient Descent adversarial block compiler with optionnal momentum.

    info: it uses the categorical_crossentropy.
    
    Params
    ------
        lr: (default=1) learning rate.
        mom: (default=0.9) momentum.

    Return
    ------
        compiler_function: a function that takes an output layer and return
            a dictionnary with :
            -train : function used to train the neural network
            -predict : function used to predict the label
            -valid : function used to get the accuracy and loss 
            -output : function used to get the output (exm: predict the label probabilities)
    
    Example:
    --------
    TODO
    """    

    concat = lasagne.layers.ConcatLayer(layers, axis=0)
    rgl = ReverseGradientLayer(concat, hp_lambda=hp_lambda)
    clf = Classifier(rgl, len(layers))
    output_layer = clf.output_layer

    input_vars = [lasagne.layers.get_all_layers(layer)[0].input_var for layer in layers]
    true_domains = [np.ones(lasagne.layers.get_all_layers(layer)[0].shape[0], dtype=np.int64)*i 
                        for i, layer in enumerate(layers)]
    true_domains = np.hstack(true_domains)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(output_layer)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, true_domains))
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent and add a momentum to it.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

    # As a bonus, also create an expression for the classification accuracy:
    acc = T.mean(T.eq(T.argmax(pred, axis=1), true_domains))
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_function = theano.function(input_vars, [loss, acc], 
        updates=updates, allow_input_downcast=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, true_domains))
    # As a bonus, also create an expression for the classification:
    label = T.argmax(pred, axis=1)
    # As a bonus, also create an expression for the classification accuracy:
    acc = T.mean(T.eq(label, true_domains))
    # Compile a second function computing the validation loss and accuracy:
    valid_function = theano.function(input_vars, [loss, acc], allow_input_downcast=True)
    # Compile a function computing the predicted labels:
    predict_function = theano.function(input_vars, [label], allow_input_downcast=True)
    # Compile an output function
    output_function = theano.function(input_vars, [pred], allow_input_downcast=True)

    funs = {
            'train': train_function,
            'predict': predict_function,
            'valid': valid_function,
            'output': output_function
           }

    return lambda ignored: funs

    