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
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier
from nn.compilers import compiler_sgd_mom
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
        default=1.0, type=float, dest='hp_lambda')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=1, type=float, dest='label_rate')
    
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
    
    # Set up the training :
    data_name = 'Moon'
    batchsize = 32
    model = 'BadNN'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # Load Moon Dataset
    source_data, target_data, domain_data = load_moon()
    datas = [source_data, domain_data, target_data]

    # Prepare the logger :
    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    shape = (None, 2)
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # Build the neural network architecture
    feature = Dense(input_layer, [5,])
    rgl = ReverseGradientLayer(feature.output_layer, hp_lambda=hp_lambda)
    label_clf = Classifier(rgl, 2)
    
    # Compilation
    logger.info('Compiling functions')
    label_trainner = Trainner(label_clf.output_layer, compiler_sgd_mom(lr=label_rate, mom=0), 'source')
    
    # Train the NN
    stats = training([label_trainner,], [source_data,],
                     num_epochs=num_epochs, logger=logger)
    
    # Plot learning accuracy curve
    fig, ax = plt.subplots()
    ax.plot(stats['source valid acc'], label='source')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0., 100.0)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('fig/'+title+'.png', bbox_inches='tight')
    fig.clf() # Clear plot window

    # Plot boundary :
    X = np.vstack([source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    plot_bound(X, y, label_trainner.output)
    plt.title('Moon bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window
    plt.close('all')


if __name__ == '__main__':
    main()
