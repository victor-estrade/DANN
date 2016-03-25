from __future__ import division, print_function

import os
import time
import abc

import theano
import lasagne

import numpy as np
import theano.tensor as T

from nn.rgl import ReverseGradientLayer
from logs import log_fname, new_logger
from utils import iterate_minibatches

class Dense(object):
    """
    A multiple dense layers.
        
    Params
    ------
        input_layer: the inputs
        arch: a list of integer giving the number of nerones in the layers.

    """
    def __init__(self, input_layer, arch):
        super(Dense, self).__init__()
        self.input_layer = input_layer
        self.arch = arch
        self.layers = []
        l_tmp = self.input_layer
        for nb_units in arch:
            l_tmp = lasagne.layers.DenseLayer(
                    l_tmp,
                    num_units=nb_units,
                    nonlinearity=lasagne.nonlinearities.tanh,
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
        arch: a list of integer giving the number of nerones in the layers.

    """
    def __init__(self, input_layer, arch, nb_label):
        super(Classifier, self).__init__()
        self.input_layer = input_layer
        self.arch = arch
        self.layers = []
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
                    num_units=nb_units,
                    nonlinearity=lasagne.nonlinearities.softmax,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )

    def copy(self, input_layer):
        copy = Dense(input_layer, self.arch)
        for layer_A, layer_copy in zip(self.layers, copy.layers):
            layer_copy.W = layer_A.W
            layer_copy.b = layer_A.b
        copy.l_output.W = self.l_output.W
        copy.l_output.b = self.l_output.b
        
        return copy

def compiler_sgd_mom(lr=1, mom=.9) : 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
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
    >>> compiler = compiler_sgd_mom(lr=0.01, mom=0.1)
    
    """    
    def get_fun(output_layer, lr=1, mom=.9, target_var=T.ivector('target')):

        input_var = lasagne.layers.get_all_layers(output_layer)[0].input_var
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        pred = lasagne.layers.get_output(output_layer)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent and add a momentum to it.
        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
        updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

        # As a bonus, also create an expression for the classification accuracy:
        acc = T.mean(T.eq(T.argmax(pred, axis=1), target_var))
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_function = theano.function([input_var, target_var], [loss, acc], 
            updates=updates, allow_input_downcast=True)
        
        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout and noise layers.
        pred = lasagne.layers.get_output(output_layer, deterministic=True)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
        # As a bonus, also create an expression for the classification:
        label = T.argmax(pred, axis=1)
        # As a bonus, also create an expression for the classification accuracy:
        acc = T.mean(T.eq(label, target_var))
        # Compile a second function computing the validation loss and accuracy:
        valid_function = theano.function([input_var, target_var], [loss, acc], allow_input_downcast=True)
        # Compile a function computing the predicted labels:
        predict_function = theano.function([input_var], [label], allow_input_downcast=True)
        # Compile an output function
        proba_function = theano.function([input_var], [pred], allow_input_downcast=True)
        
        return {
                'train': train_function,
                'predict': predict_function,
                'valid': valid_function,
                'output': proba_function
               }
    
    return lambda output_layer: get_fun(output_layer, lr=lr, mom=mom)
