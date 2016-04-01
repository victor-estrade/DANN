#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import theano
import theano.tensor as T
import lasagne

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from datasets.moon import load_moon
from datasets.mnist import load_mnist_mirror
from datasets.utils import random_mat_dataset
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier
from nn.compilers import compiler_sgd_mom
from nn.training import Trainner, training
from utils import plot_bound


def squared_error_sgd_mom(lr=1, mom=.9, target_var=T.ivector('target')) : 
    """
    Stochastic Gradient Descent compiler with optionnal momentum.

    info: it uses the squared_error.
    
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
        loss = T.mean(lasagne.objectives.squared_error(pred, target_var))
        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent and add a momentum to it.
        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
        updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)

        # As a bonus, also create an expression for the classification accuracy:
        acc = T.mean((pred - target_var)**2)
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_function = theano.function([input_var, target_var], [loss, acc], 
            updates=updates, allow_input_downcast=True)
        
        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout and noise layers.
        pred = lasagne.layers.get_output(output_layer, deterministic=True)
        loss = T.mean(lasagne.objectives.squared_error(pred, target_var))
        # As a bonus, also create an expression for the classification:
        label = T.argmax(pred, axis=1)
        # As a bonus, also create an expression for the classification accuracy:
        acc = T.mean((pred - target_var)**2)
        # Compile a second function computing the validation loss and accuracy:
        valid_function = theano.function([input_var, target_var], [loss, acc], allow_input_downcast=True)
        # Compile a function computing the predicted labels:
        predict_function = theano.function([input_var], [label], allow_input_downcast=True)
        # Compile an output function
        output_function = theano.function([input_var], [pred], allow_input_downcast=True)
        
        return {
                'train': train_function,
                'predict': predict_function,
                'valid': valid_function,
                'output': output_function
               }
    
    return lambda output_layer: get_fun(output_layer, lr=lr, mom=mom, target_var=target_var)


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
        '--epoch', help='Number of epoch in the training session',
        default=50, type=int, dest='num_epochs')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0.7, type=float, dest='hp_lambda')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=1, type=float, dest='label_rate')
    parser.add_argument(
        '--domain-rate', help="The learning rate of the domain part of the neural network ",
        default=1, type=float, dest='domain_rate')

    args = parser.parse_args()
    return args


def main():
    """
    The main function.
    """
    # Parse the aruments
    args = parseArgs()
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    label_rate = args.label_rate
    domain_rate = args.domain_rate

    # Set up the training :
    data_name = 'MoonRMat'
    batchsize = 32
    model = 'SimplestCorrector'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # Load Moon Dataset
    # source_data, target_data, domain_data = load_moon()
    # source_data, target_data, domain_data = random_mat_dataset(source_data)
    
    # Load MNIST Dataset
    source_data, target_data, domain_data = load_mnist_mirror()
    
    corrector_data = dict(target_data)
    corrector_data.update({
    	'y_train':source_data['X_train'],
    	'y_val':source_data['X_val'],
    	'y_test':source_data['X_test'],
    	})

    # Prepare the logger :
    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))

    # Prepare Theano variables for inputs and targets
    # input_var = T.matrix('inputs')
    # target_var = T.matrix('targets')
    # shape = (None, 2)
    input_var = T.tensor3('inputs')
    target_var = T.tensor3('targets')
    shape = (None, 28, 28)
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    #=========================================================================
    # Build the neural network architecture
    #=========================================================================
    feature = lasagne.layers.DenseLayer(
                    input_layer,
                    num_units=np.prod(shape[1:]),
                    nonlinearity=None,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
    reshaper = lasagne.layers.ReshapeLayer(feature, (-1,) + shape[1:])
    output_layer = reshaper
    
    # Compilation
    logger.info('Compiling functions')
    corrector_trainner = Trainner(output_layer, squared_error_sgd_mom(lr=label_rate, mom=0, target_var=target_var), 
    							 'corrector',)
    # domain_trainner = Trainner(domain_clf.output_layer, squared_error_sgd_mom(lr=domain_rate, mom=0), 'domain')
    # target_trainner = Trainner(label_clf.output_layer, squared_error_sgd_mom(lr=label_rate, mom=0), 'target')

    # Train the NN
    stats = training([corrector_trainner,], [corrector_data,],
                     # testers=[target_trainner,], test_data=[target_data],
                     num_epochs=num_epochs, logger=logger)
    
    # Plot learning accuracy curve
    fig, ax = plt.subplots()
    ax.plot(stats['corrector valid loss'], label='source')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('fig/'+title+'.png', bbox_inches='tight')
    fig.clf() # Clear plot window


if __name__ == '__main__':
    main()
