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

from datasets import load_moon
from rgl import ReverseGradientLayer
from logs import log_fname, new_logger
from compilers import compiler_sgd_mom
from utils import plot_bound
from dann import ShallowDANN


def main(hp_lambda=0.0, num_epochs=50, angle=-35, label_rate=1, domain_rate=1):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'moon'
    batchsize = 32
    source_data, target_data, domain_data = load_moon(angle=angle)

    # Set up the training :
    datas = [source_data, domain_data, target_data]

    model = '1DR'

    title = '{}-lambda-{:.4f}-{}'.format(model, hp_lambda, data_name)
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
    dann = ShallowDANN(3, 2, input_layer, hp_lambda=hp_lambda)

    logger.info('Compiling functions')
    dann.compile_label(compiler_sgd_mom(lr=label_rate, mom=0))
    dann.compile_domain(compiler_sgd_mom(lr=domain_rate, mom=0))

    # Train the NN
    dann.training(source_data, domain_data, num_epochs=num_epochs)

    # Plot boundary :
    X = np.vstack([source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, dann.proba_label)
    plt.title('Moon bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window

    X = np.vstack([target_data['X_train'], target_data['X_val'], target_data['X_test'], ])
    y = np.hstack([target_data['y_train'], target_data['y_val'], target_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, dann.proba_label)
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
        '--epoch', help='Number of epoch in the training session',
        default=50, type=int, dest='num_epochs')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0.7, type=float, dest='hp_lambda')
    parser.add_argument(
        '--angle', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=-35., type=float, dest='angle')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=1, type=float, dest='label_rate')
    parser.add_argument(
        '--domain-rate', help="The learning rate of the domain part of the neural network ",
        default=1, type=float, dest='domain_rate')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    angle = args.angle
    label_rate = args.label_rate
    domain_rate = args.domain_rate
    main(hp_lambda=hp_lambda,  num_epochs=num_epochs, angle=angle,
        label_rate=label_rate, domain_rate=domain_rate,)
