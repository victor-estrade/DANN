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
