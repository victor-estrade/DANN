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
from datasets.utils import random_mat_dataset
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier
from nn.compilers import squared_error_sgd_mom
from nn.training import Trainner, training
from utils import plot_bound


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
        default=12, type=int, dest='num_epochs')
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
    model = 'PaiwiseCorrector'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # Load Moon Dataset
    source_data, target_data, domain_data = load_moon()
    source_data, target_data, domain_data = random_mat_dataset(source_data)
    
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
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    shape = (None, 2)
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    #=========================================================================
    # Build the neural network architecture
    #=========================================================================
    output_layer = lasagne.layers.DenseLayer(
                    input_layer,
                    num_units=np.prod(shape[1:]),
                    nonlinearity=None,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )

    # Compilation
    logger.info('Compiling functions')
    corrector_trainner = Trainner(output_layer, squared_error_sgd_mom(lr=label_rate, mom=0, target_var=target_var), 
    							 'corrector',)

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

    # Plot the source, target and corrected data
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    color = cm.ScalarMappable(cmap=cm_bright)
    
    fig, ax = plt.subplots()
    X = source_data['X_test']
    y = source_data['y_test']
    ax.scatter(X[:, 0], X[:, 1], label='source', marker='o', s=80, edgecolors=color.to_rgba(y), facecolors='none')

    X = target_data['X_test']
    y = target_data['y_test']
    ax.scatter(X[:, 0], X[:, 1], label='target', marker='D', s=80, edgecolors=color.to_rgba(y), facecolors='none')
    
    X = np.array(corrector_trainner.output(target_data['X_test'])).reshape((-1, 2))
    y = target_data['y_test']
    ax.scatter(X[:, 0], X[:, 1], label='corrected', marker='x', s=80, c=y, cmap=cm_bright)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('fig/'+title+'.png', bbox_inches='tight')

if __name__ == '__main__':
    main()