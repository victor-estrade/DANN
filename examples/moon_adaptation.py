#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os

import theano
import lasagne
import argparse

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from datasets import load_moon
# from nn_compilers import compile_sgd
from logs import log_fname, new_logger
from run import Path, training, plot_bound
from rgl import ReverseGradientLayer

def build_nn(input_var=None, hp_lambda=0.5, shape=(None, 2)):
    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)

    # A fully-connected layer of 256 units
    feature = lasagne.layers.DenseLayer(
            input_layer,
            num_units=3,
            nonlinearity=lasagne.nonlinearities.tanh,
            # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
            )

    # Reversal gradient layer
    RGL = ReverseGradientLayer(feature, hp_lambda=hp_lambda)
    
    # Label classifier
    label_predictor = lasagne.layers.DenseLayer(
            feature,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )
    domain_predictor = lasagne.layers.DenseLayer(
            RGL,
            # domain_hidden,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor


def compile_sgd(nn, input_var=None, target_var=None, learning_rate=1):
    """
    Compile the given path of a neural network.
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


def main(hp_lambda=0.0, angle=-35):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'moon'
    batchsize = 32
    source_data, target_data, domain_data = load_moon(angle=angle)
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    shape = (None, 2)

    # Set up the training :
    datas = [source_data, domain_data, target_data]

    model = '1DR'

    title = '{}-lambda-{:.4f}-{}'.format(model, hp_lambda, data_name)
    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))
    
    # Build the neural network architecture
    label_nn, domain_nn = build_nn(input_var=input_var, shape=shape, hp_lambda=hp_lambda)
    label_path = Path(label_nn, compile_sgd, input_var=input_var,
            target_var=target_var, name='source')
    domain_path = Path(domain_nn, compile_sgd, input_var=input_var,
            target_var=target_var, name='source')
    target_path = Path(label_nn, compile_sgd, input_var=input_var,
            target_var=target_var, name='source', trainable=False)
    pathes = [label_path, domain_path, target_path]

    # Train the NN
    training(datas, pathes, num_epochs=50)

    # Plot learning accuracy curve
    fig0, ax0 = plt.subplots()
    ax0.plot(label_path.val_stats['acc'], label='source', c='blue')
    ax0.plot(target_path.val_stats['acc'], label='target', c='red')
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('accuracy')
    ax0.set_ylim(0., 100.0)
    ax0.set_title(title)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig0.savefig('fig/'+title+'.png', bbox_inches='tight')
    fig0.clf() # Clear plot window
    
    # Plot boundary :
    X = np.vstack([source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, label_path.output_fn)
    plt.title('Moon bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window
    
    X = np.vstack([target_data['X_train'], target_data['X_val'], target_data['X_test'], ])
    y = np.hstack([target_data['y_train'], target_data['y_val'], target_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, label_path.output_fn)
    plt.title('Moon rotated bounds')
    plt.savefig('fig/moon-rot-bound.png')
    plt.clf() # Clear plot window
    
def parseArgs():
    """
    ArgumentParser.

    Return
    ------
        args: the parsed arguments.
    """
    # Retrieve the arguments
    parser = argparse.ArgumentParser(
        description="Reverse gradient example -- Example of the destructive"
                    "power of the Reverse Gradient Layer")
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0, type=float, dest='hp_lambda')
    parser.add_argument(
        '--angle', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=-35., type=float, dest='angle')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    hp_lambda = args.hp_lambda
    angle = args.angle
    main(hp_lambda=hp_lambda, angle=angle)
