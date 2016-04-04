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
        default=50, type=int, dest='num_epochs')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0., type=float, dest='hp_lambda')
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
    data_name = 'MNISTMirror'
    batchsize = 32
    model = 'SimplestCorrector'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # Load MNIST Dataset
    source_data, target_data, domain_data = load_mnist_mirror()
    
    corrector_data = dict(target_data)
    corrector_data.update({
    	'y_train': source_data['X_train'],
    	'y_val': source_data['X_val'],
    	'y_test': source_data['X_test'],
    	})

    # Prepare the logger :
    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))

    # Prepare Theano variables for inputs and targets
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
    
    # Train the NN
    stats = training([corrector_trainner,], [corrector_data,],
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

    # Plot some sample images:
    fig = plt.figure()
    n_sample = 4
    rand = np.random.RandomState()
    for n in range(n_sample):
        i = rand.randint(source_data['X_test'].shape[0])
        sample_src = source_data['X_test'][i]
        sample_trg = target_data['X_test'][i]
        sample_corrected = corrector_trainner.output(target_data['X_test'][i][np.newaxis])
        sample_corrected = np.array(sample_corrected).reshape((28,28))
        ax = fig.add_subplot(n_sample, 3, n*3+1)
        ax.axis('off')
        ax.imshow(sample_src, cmap='Greys_r')
        ax.set_title('Source image')
        ax = fig.add_subplot(n_sample, 3, n*3+2)
        ax.axis('off')
        ax.imshow(sample_trg, cmap='Greys_r')
        ax.set_title('Target image')
        ax = fig.add_subplot(n_sample, 3, n*3+3)
        ax.axis('off')
        ax.imshow(sample_corrected, cmap='Greys_r')
        ax.set_title('Corrected image')
    fig.savefig('fig/{}-sample.png'.format(title))
    plt.close(fig) # Clear plot window


if __name__ == '__main__':
    main()
