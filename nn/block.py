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


class Dense(object):
    """
    A multiple dense layers.
        
    Params
    ------
        input_layer: the inputs
        arch: a list of integer giving the number of nerones in the layers.
        nonlinearity: (default=tanh) the nonlinearity activation on every layers.

    """
    def __init__(self, input_layer, arch, nonlinearity=lasagne.nonlinearities.tanh):
        super(Dense, self).__init__()
        self.input_layer = input_layer
        self.arch = arch
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
            
        self.l_output = l_tmp

    def copy(self, input_layer):
        copy = Dense(input_layer, self.arch)
        for layer_A, layer_copy in zip(self.layers, copy.layers):
            layer_copy.W = layer_A.W
            layer_copy.b = layer_A.b
        return copy



class Classifier(object):
    """
    A multiple dense layers ending by a classifier.
        
    Params
    ------
        input_layer: the inputs
        nb_label : the number of label to be predicted. 
            (number of neurone in the last softmax layer)
        arch: (default=[]) a list of integer giving the number of nerones in the layers.

    """
    def __init__(self, input_layer, nb_label, arch=[]):
        super(Classifier, self).__init__()
        self.input_layer = input_layer
        self.arch = arch
        self.layers = []
        self.nb_label = nb_label
        
        l_tmp = self.input_layer
        for nb_units in arch:
            l_tmp = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_units,
                    nonlinearity=lasagne.nonlinearities.tanh,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
            self.layers.append(l_tmp)
        self.l_output = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_label,
                    nonlinearity=lasagne.nonlinearities.softmax,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )

    def copy(self, input_layer):
        copy = Classifier(input_layer, self.nb_label, self.arch)
        for layer_A, layer_copy in zip(self.layers, copy.layers):
            layer_copy.W = layer_A.W
            layer_copy.b = layer_A.b
        copy.l_output.W = self.l_output.W
        copy.l_output.b = self.l_output.b
        
        return copy
