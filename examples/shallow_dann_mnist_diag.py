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

from datasets.mnist import load_mnist_src
from datasets.utils import diag_dataset
from nn.dann import ShallowDANN
from nn.compilers import compiler_sgd_mom
from logs import log_fname, new_logger
from utils import plot_bound


def main(hp_lambda=0.1, num_epochs=50, label_rate=1, domain_rate=1):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'MnistA'
    batchsize = 500
    source_data = load_mnist_src()
    source_data, target_data, domain_data = diag_dataset(source_data)
    datas = [source_data, domain_data, target_data]
    # Set up the training :
    model = 'ShallowDANN'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Data: {}'.format(data_name))
    logger.info('Model: {}'.format(model))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')
    shape = (None, 28, 28)
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # Build the neural network architecture
    dann = ShallowDANN(50, 10, input_layer, hp_lambda=hp_lambda)

    logger.info('Compiling functions')
    dann.compile_label(compiler_sgd_mom(lr=label_rate, mom=0))
    dann.compile_domain(compiler_sgd_mom(lr=domain_rate, mom=0))

    # Train the NN
    stats = dann.training(source_data, domain_data, 
        target=target_data, num_epochs=num_epochs, logger=logger)

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

    # Sample image:
    i = np.random.RandomState().randint(source_data['X_test'].shape[0])
    sample_src = source_data['X_test'][i]
    sample_trg = target_data['X_test'][i]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample_src, cmap='Greys_r')
    label = dann.predict_label(sample_src[np.newaxis])[0]
    ax.set_title('Source image (pred={})'.format(label))
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(sample_trg, cmap='Greys_r')
    label = dann.predict_label(sample_trg[np.newaxis])[0]
    ax.set_title('Target image (pred={})'.format(label))
    fig.savefig('fig/MNIST-sample.png')
    fig.clf() # Clear plot window
    

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
        default=0.01, type=float, dest='hp_lambda')
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
    label_rate = args.label_rate
    domain_rate = args.domain_rate
    main(hp_lambda=hp_lambda,  num_epochs=num_epochs,
        label_rate=label_rate, domain_rate=domain_rate,)
