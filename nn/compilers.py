#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import theano
import lasagne

import numpy as np
import theano.tensor as T

from rgl import ReverseGradientLayer



def get_input_vars(out_layer):
    """
    Get the all input variables the out_layer depends on.
    
    Params
    -----
        out_layer: the layer we search input variables on.
    Return
    ------
        input_vars: the input variables.
    """
    layers = lasagne.layers.get_all_layers(out_layer)
    input_vars = [layer.input_var for layer in layers if isinstance(layer, lasagne.layers.InputLayer)]
    return input_vars


def classification_sgd_mom(output_layer, lr=1, mom=.9, target_var=T.ivector('target'),
                        regularization=None, reg_param=0.1): 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed
        target_var : (default=T.ivector('target')) the target symbolic variable
        lr: (default=1) learning rate.
        mom: (default=0.9) momentum.
        regularisation: (default=None) the regularization, can be 'l1' or 'l2' or None.
        reg_param: (default=0.1) the regularization hyper parameter: 
                        loss = loss + reg_param * regularization

    Return
    ------
        A dictionnary with :
            -train : function used to train the neural network (same as fit)
            -train_desription: ('loss', 'acc'),
            -fit : function used to train the neural network (same as train)
            -fit_description: ('loss', 'acc'),

    Example:
    --------
    >>> funs = classification_sgd_mom(output_layer, lr=0.01, mom=0.1)
    >>> loss, acc = funs['train'](X, y)
    """
    input_vars = get_input_vars(output_layer)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(output_layer)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
    # Add a regularization term to the loss if needed
    if regularization == 'l1':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l1)
        loss += reg_param*reg
    elif regularization == 'l2':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss += reg_param*reg
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
    train_function = theano.function(input_vars+[target_var], [loss, acc], 
        updates=updates, allow_input_downcast=True)

    return {
            'train': train_function,
            'train_description': ('loss', 'acc'),
            'fit': train_function,
            'fit_description': ('loss', 'acc'),
            }


def crossentropy_sgd_mom(output_layer, lr=1, mom=.9, target_var=T.matrix('target'),
                        regularization=None, reg_param=0.1): 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed
        target_var : (default=T.matrix('target')) the target symbolic variable
        lr: (default=1) learning rate.
        mom: (default=0.9) momentum.
        regularisation: (default=None) the regularization, can be 'l1' or 'l2' or None.
        reg_param: (default=0.1) the regularization hyper parameter: 
                        loss = loss + reg_param * regularization

    Return
    ------
        A dictionnary with :
            -train : function used to train the neural network (same as fit)
            -train_desription: ('loss',),
            -fit : function used to train the neural network (same as train)
            -fit_description: ('loss',),

    Example:
    --------
    >>> funs = crossentropy_sgd_mom(output_layer, lr=0.01, mom=0.1)
    >>> loss = funs['train'](X, y)
    
    """

    input_vars = get_input_vars(output_layer)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(output_layer)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
    # Add a regularization term to the loss if needed
    if regularization == 'l1':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l1)
        loss += reg_param*reg
    elif regularization == 'l2':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss += reg_param*reg
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent and add a momentum to it.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_function = theano.function(input_vars+[target_var], [loss], 
        updates=updates, allow_input_downcast=True)

    return {
            'train': train_function,
            'train_description': ('loss',),
            'fit': train_function,
            'fit_description': ('loss',),
            }

def crossentropy_validation(output_layer, target_var=T.matrix('target')):
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed
        target_var : (default=T.matrix('target')) the target symbolic variable

    Return
    ------
        A dictionnary with :
            -valid : function used to get the accuracy and loss 
            -valid_description: ('loss',),

    Example:
    --------
    >>> funs = crossentropy_validation(output_layer)
    >>> loss = funs['valid'](X, y)

    """    
    input_vars = get_input_vars(output_layer)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
    # Compile a second function computing the validation loss and accuracy:
    valid_function = theano.function(input_vars+[target_var], [loss,], allow_input_downcast=True)
    return {
            'valid': valid_function,
            'valid_description': ('loss',),
           }


def classification_validation(output_layer, target_var=T.ivector('target')):
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed
        target_var : (default=T.ivector('target')) the target symbolic variable

    Return
    ------
        A dictionnary with :
            -valid : function used to get the accuracy and loss 
            -valid_description: ('loss', 'acc'),

    Example:
    --------
    >>> funs = classification_validation(output_layer)
    >>> loss, acc = funs['valid'](X, y)

    """    
    input_vars = get_input_vars(output_layer)
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
    valid_function = theano.function(input_vars+[target_var], [loss, acc], allow_input_downcast=True)

    return {
            'valid': valid_function,
            'valid_description': ('loss', 'acc'),
           }


def argmax_prediction(output_layer):
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed

    Return
    ------
        A dictionnary with :
            -predict : function used to predict the label
            -predict_description: ('label'),

    Example:
    --------
    >>> funs = argmax_prediction(output_layer)
    >>> labels = funs['argmax'](X)
    """    
    input_vars = get_input_vars(output_layer)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    # As a bonus, also create an expression for the classification:
    label = T.argmax(pred, axis=1)
    # Compile a function computing the predicted labels:
    predict_function = theano.function(input_vars, [label], allow_input_downcast=True)

    return {
            'predict': predict_function,
            'predict_description': ('label'),
           }


def output(output_layer):
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the categorical_crossentropy. Should be given to a softmax layer.
    
    Params
    ------
        output_layer: the output layer from which the loss and updates will be computed

    Return
    ------
        A dictionnary with :
            -output : function used to get the output (exm: predict the label probabilities)
            -output_description: ('output'),

    Example:
    --------
    >>> funs = output(output_layer)
    >>> raw_output = funs['output'](X)

    """    
    input_vars = get_input_vars(output_layer)
    # Create an expression for the output of the layer. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    # Compile an output function
    output_function = theano.function(input_vars, [pred], allow_input_downcast=True)

    return {
            'output': output_function,
            'output_description': ('output'),
           }


def squared_error_sgd_mom(output_layer, lr=1, mom=.9, target_var=T.matrix('target'),
                        regularization=None, reg_param=0.1): 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the squared_error.
    
    Params
    ------
        output_layer: the output layer from which the loss and updtaes will be computed
        target_var : (default=T.matrix('target')) the target symbolic variable
        lr: (default=1) learning rate.
        mom: (default=0.9) momentum.
        regularisation: (default=None) the regularization, can be 'l1' or 'l2' or None.
        reg_param: (default=0.1) the regularization hyper parameter: 
                        loss = loss + reg_param * regularization

    Return
    ------
        A dictionnary with :
            -train : function used to train the neural network (same as fit)
            -train_description: ('loss',),
            -fit : function used to train the neural network (same as train)
            -fit_description: ('loss',),

    Example:
    --------
    >>> funs = squared_error_sgd_mom(output_layer, lr=0.01, mom=0.1)
    >>> loss = funs['train'](X, y)
    
    """
    input_vars = get_input_vars(output_layer)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(output_layer)
    loss = T.mean(lasagne.objectives.squared_error(pred, target_var))
    # Add a regularization term to the loss if needed
    if regularization == 'l1':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l1)
        loss += reg_param*reg
    elif regularization == 'l2':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss += reg_param*reg
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent and add a momentum to it.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_function = theano.function(input_vars+[target_var], [loss], 
        updates=updates, allow_input_downcast=True)

    return {
            'train': train_function,
            'train_description': ('loss',),
            'fit': train_function,
            'fit_description': ('loss',),
           }


def squared_error_validation(output_layer, target_var=T.matrix('target')): 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the squared_error.
    
    Params
    ------
        output_layer: the output layer from which the loss and updtaes will be computed
        target_var : (default=T.matrix('target')) the target symbolic variable

    Return
    ------
        A dictionnary with :
            -valid : function used to get the accuracy and loss 
            -valid_description: ('loss',),

    Example:
    --------
    >>> funs = squared_error_validation(output_layer)
    >>> loss = funs['valid'](X, y)

    """
    input_vars = get_input_vars(output_layer)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    loss = T.mean(lasagne.objectives.squared_error(pred, target_var))
    # Compile a second function computing the validation loss and accuracy:
    valid_function = theano.function(input_vars+[target_var], [loss,], allow_input_downcast=True)
    
    return {
            'valid': valid_function,
            'valid_description': ('loss',),
           }


def adversarial(layers, hp_lambda=1, lr=1, mom=.9,
                        regularization=None, reg_param=0.1): 
    """
    Stochastic Gradient Descent adversarial block compiler with optionnal momentum.

    info: it uses the categorical_crossentropy.
    
    Params
    ------
        layers: list of output layers, one for each domain.
        lr: (default=1) learning rate.
        mom: (default=0.9) momentum.
        regularisation: (default=None) the regularization, can be 'l1' or 'l2' or None.
        reg_param: (default=0.1) the regularization hyper parameter: 
                        loss = loss + reg_param * regularization

    Return
    ------
        compiler_function: a function that takes an output layer and return
            a dictionnary with :
            -train : function used to train the neural network (same as fit)
            -train_description: ('loss', domain_acc_0, ..., domain_acc_n ),
            -fit : function used to train the neural network (same as train)
            -fit_description: ('loss', domain_acc_0, ..., domain_acc_n ),
            -predict : function used to predict the label
            -predict_description: ('label'),
            -valid : function used to get the accuracy and loss 
            -valid_description: ('loss', domain_acc_0, ..., domain_acc_n ),
            -output : function used to get the output (exm: predict the label probabilities)
            -output_description: ('prediction'),

    Example:
    --------
    TODO
    """

    concat = lasagne.layers.ConcatLayer(layers, axis=0)
    rgl = ReverseGradientLayer(concat, hp_lambda=hp_lambda)
    output_layer = lasagne.layers.DenseLayer(
                    rgl,
                    num_units=len(layers),
                    nonlinearity=lasagne.nonlinearities.softmax,
                    )

    input_vars = get_input_vars(output_layer)
    true_domains = [np.ones(lasagne.layers.get_all_layers(layer)[0].shape[0], dtype=np.int32)*i 
                        for i, layer in enumerate(layers)]
    true_domains = np.hstack(true_domains)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(output_layer)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, true_domains))
    # Add a regularization term to the loss if needed
    if regularization == 'l1':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l1)
        loss += reg_param*reg
    elif regularization == 'l2':
        reg = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss += reg_param*reg
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent and add a momentum to it.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

    # As a bonus, also create an expression for the classification accuracy:
    n_samples = np.cumsum([0]+[lasagne.layers.get_all_layers(layer)[0].shape[0] for layer in layers])
    accs = [T.mean(T.eq(T.argmax(pred[n:m], axis=1), true_domains[n:m])) 
            for n, m in zip(n_samples[:-1],n_samples[1:])]
    # acc = T.mean(T.eq(T.argmax(pred, axis=1), true_domains))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_function = theano.function(input_vars, [loss,]+accs, 
        updates=updates, allow_input_downcast=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout and noise layers.
    pred = lasagne.layers.get_output(output_layer, deterministic=True)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, true_domains))
    # As a bonus, also create an expression for the classification:
    label = T.argmax(pred, axis=1)
    # As a bonus, also create an expression for the classification accuracy:
    # n_samples = np.cumsum([0]+[lasagne.layers.get_all_layers(layer)[0].shape[0] for layer in layers])
    accs = [T.mean(T.eq(T.argmax(pred[n:m], axis=1), true_domains[n:m])) 
            for n, m in zip(n_samples[:-1],n_samples[1:])]
    # acc = T.mean(T.eq(label, true_domains))

    # Compile a second function computing the validation loss and accuracy:
    valid_function = theano.function(input_vars, [loss,]+accs, allow_input_downcast=True)
    # Compile a function computing the predicted labels:
    predict_function = theano.function(input_vars, [label], allow_input_downcast=True)
    # Compile an output function
    output_function = theano.function(input_vars, [pred], allow_input_downcast=True)

    desc = ['acc_domain_'+str(i) for i in range(len(accs))]
    funs = {
            'train': train_function,
            'train_description': ('loss',)+tuple(desc),
            'fit': train_function,
            'fit_description': ('loss',)+tuple(desc),
            'predict': predict_function,
            'predict_description': ('label'),
            'valid': valid_function,
            'valid_description': ('loss',)+tuple(desc),
            'output': output_function,
            'output_description': ('prediction'),
           }

    return funs

