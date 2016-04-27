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

from datasets.toys import load_moons
from datasets.transform import diag_dataset
from datasets.utils import make_domain_dataset
from logs import log_fname, new_logger
from nn.clone import clone_layer
from nn.compilers import crossentropy_sgd_mom, adversarial
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
        description="Moon Diag adaptation example")
    parser.add_argument(
        '--epoch', help='Number of epoch in the training session',
        default=100, type=int, dest='num_epochs')
    parser.add_argument(
        '--batchsize', help='The mini-batch size',
        default=32, type=int, dest='batchsize')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0.8, type=float, dest='hp_lambda')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=1, type=float, dest='label_rate')
    parser.add_argument(
        '--label-mom', help="The learning rate momentum of the label part of the neural network ",
        default=0.9, type=float, dest='label_mom')
    parser.add_argument(
        '--domain-rate', help="The learning rate of the domain part of the neural network ",
        default=1, type=float, dest='domain_rate')
    parser.add_argument(
        '--domain-mom', help="The learning rate momentum of the domain part of the neural network ",
        default=0.9, type=float, dest='domain_mom')

    args = parser.parse_args()
    return args


def main():
    """
    The main function.
    """
    #=========================================================================
    # Parse the arguments. Handle the parameters
    #=========================================================================
    args = parseArgs()
    num_epochs = args.num_epochs
    batchsize = args.batchsize
    hp_lambda = args.hp_lambda
    label_rate = args.label_rate
    label_mom = args.label_mom
    domain_rate = args.domain_rate
    domain_mom = args.domain_mom

    # Set up the naming information :
    data_name = 'MoonDiag'
    model = 'SimplestDANN'
    title = '{}-{}-lambda-{:.2e}'.format(data_name, model, hp_lambda)

    #=========================================================================
    # Load, Generate the datasets
    #=========================================================================
    # Load Moon Dataset
    source_data = load_moons(batchsize=batchsize)
    target_data = diag_dataset(source_data)
    domain_data = make_domain_dataset([source_data, target_data])

    #=========================================================================
    # Prepare the logger
    #=========================================================================
    # f_log = log_fname(title)
    logger = new_logger()
    # Print general information
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('Batchsize: {}'.format(batchsize))
    logger.info('hp_lambda = {:.4e}'.format(hp_lambda))

    #=========================================================================
    # Build the neural network architecture
    #=========================================================================
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    shape = (batchsize, 2)

    # Build the layers
    input_layer = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    feature = lasagne.layers.DenseLayer(
                input_layer,
                num_units=5,
                nonlinearity=lasagne.nonlinearities.tanh,
                )
    label_layer = lasagne.layers.DenseLayer(
                feature,
                num_units=2,
                nonlinearity=lasagne.nonlinearities.softmax,
                )
    
    input_layer2 = lasagne.layers.InputLayer(shape=shape, input_var=T.matrix('inputs2'))
    feature2 = clone_layer(feature, input_layer2)
    #label_layer2 = clone_layer(label_layer, feature)
    
    # Compilation
    logger.info('Compiling functions')
    label_trainner = Trainner(crossentropy_sgd_mom(label_layer, lr=label_rate, mom=label_mom), 'source')
    domain_trainner = Trainner(adversarial([feature, feature2], hp_lambda=hp_lambda, lr=domain_rate, mom=domain_mom),'domain')
    target_tester = Trainner(crossentropy_sgd_mom(label_layer, lr=label_rate, mom=label_mom), 'target')

    #=========================================================================
    # Train the Neural Network
    #=========================================================================
    stats = training([label_trainner, domain_trainner], [source_data, domain_data],
                     testers=[target_tester,], test_data=[target_data],
                     num_epochs=num_epochs, logger=logger)
    
    #=========================================================================
    # Print, Plot, Save the final results
    #=========================================================================
    # Plot learning accuracy curve
    fig, ax = plt.subplots()
    ax.plot(stats['source valid acc'], label='source')
    ax.plot(stats['target valid acc'], label='target')
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
    plt.title('Moon Diag bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window

    X = np.vstack([target_data['X_train'], target_data['X_val'], target_data['X_test'], ])
    y = np.hstack([target_data['y_train'], target_data['y_val'], target_data['y_test'], ])
    plot_bound(X, y, label_trainner.output)
    plt.title('Moon Diag bounds')
    plt.savefig('fig/moon-diag-bound.png')
    plt.clf() # Clear plot window

    X = np.vstack([target_data['X_train'], target_data['X_val'], target_data['X_test'],
                    source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([target_data['y_train'], target_data['y_val'], target_data['y_test'],
                    source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    plot_bound(X, y, label_trainner.output)
    plt.title('Moon Diag Mix bounds')
    plt.savefig('fig/moon-diag-mix-bound.png')
    plt.clf() # Clear plot window


if __name__ == '__main__':
    main()
