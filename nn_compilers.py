#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import theano
import lasagne

import theano.tensor as T


def compile_sgd(nn, input_var=None, target_var=None, learning_rate=0.01):
    """Compile the given path of a neural network.
    """
    if input_var is None:
        input_var = lasagne.layers.get_all_layers(nn)[0].input_var
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    train_output = lasagne.layers.get_output(nn)
    train_loss = lasagne.objectives.categorical_crossentropy(train_output, target_var)
    train_loss = train_loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(nn, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         train_loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.sgd(train_loss, params, learning_rate=learning_rate)

    # As a bonus, also create an expression for the classification accuracy:
    train_acc = T.mean(T.eq(T.argmax(train_output, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [train_loss, train_acc], updates=updates,
                               allow_input_downcast=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_output = lasagne.layers.get_output(nn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_output,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],
                             allow_input_downcast=True)
    # Compile an output function
    output_fn = theano.function([input_var],
                                [test_output],
                                allow_input_downcast=True)
    return train_fn, val_fn, output_fn


def compile_nesterov(nn, input_var=None, target_var=None, learning_rate=0.01,
                     momentum=0.9):
    """Compile the given path of a neural network.
    """
    if input_var is None:
        input_var = lasagne.layers.get_all_layers(nn)[0].input_var
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    train_output = lasagne.layers.get_output(nn)
    train_loss = lasagne.objectives.categorical_crossentropy(train_output, target_var)
    train_loss = train_loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(nn, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            train_loss, params, learning_rate=learning_rate, momentum=momentum)
    
    # As a bonus, also create an expression for the classification accuracy:
    train_acc = T.mean(T.eq(T.argmax(train_output, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [train_loss, train_acc], updates=updates,
                               allow_input_downcast=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_output = lasagne.layers.get_output(nn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_output,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],
                             allow_input_downcast=True)
    # Compile an output function
    output_fn = theano.function([input_var],
                                [test_output],
                                allow_input_downcast=True)
    return train_fn, val_fn, output_fn
