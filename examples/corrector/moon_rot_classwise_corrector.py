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

from datasets.toys import load_moon
from datasets.utils import rotate_dataset, make_domain_dataset
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier, adversarial
from nn.compilers import squared_error_sgd_mom, crossentropy_sgd_mom
from nn.training import Trainner, training
from utils import plot_bound, iterate_minibatches


def classwise_shuffle(X, y):
    """
    Shuffle X without changing the class positions

    Params
    ------
        X: the data (numpy array)
        y: the labels 
    Return
    ------
        X_shuffled: Shuffled X without changing the class matching
    """
    idx = np.empty_like(y, dtype=int)
    for label in np.unique(y):
        arr = np.where(y==label)[0]
        arr2 = np.random.permutation(arr)
        idx[arr] = arr2
    return X[idx]


def epoch_shuffle(data, trainer, epoch, *args, **kwargs):
    data['X_train'] = classwise_shuffle(data['X_train'], data['labels'])
    return data


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
        default=40, type=int, dest='num_epochs')
    parser.add_argument(
        '--angle', help='Angle of the rotation applied to the datasets',
        default=-30., type=float, dest='angle')
    parser.add_argument(
        '--batchsize', help='The mini-batch size',
        default=32, type=int, dest='batchsize')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0., type=float, dest='hp_lambda')
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
        default=0., type=float, dest='domain_mom')

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
    angle = args.angle
    batchsize = args.batchsize
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    label_rate = args.label_rate
    label_mom = args.label_mom
    domain_rate = args.domain_rate
    domain_mom = args.domain_mom

    # Set up the naming information :
    data_name = 'MoonRotated'
    model = 'ClassWiseCorrector'
    title = '{}-{}-lambda-{:.2e}'.format(data_name, model, hp_lambda)

    #=========================================================================
    # Load, Generate the datasets
    #=========================================================================
    # Load Moon Dataset
    source_data = load_moon(batchsize=batchsize)
    target_data = rotate_dataset(source_data, angle=angle)
    domain_data = make_domain_dataset([source_data, target_data])

    corrector_data = dict(target_data)
    corrector_data.update({
        'y_train':source_data['X_train'],
        'y_val':source_data['X_val'],
        'y_test':source_data['X_test'],
        'labels': source_data['y_train'],
        'batchsize': batchsize,
        })

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
    target_var = T.matrix('targets')
    shape = (batchsize, 2)

    # Build the layers
    input_layer = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    src_layer = lasagne.layers.InputLayer(shape=shape, input_var=T.matrix('src'))
    output_layer = lasagne.layers.DenseLayer(
                    input_layer,
                    num_units=np.prod(shape[1:]),
                    nonlinearity=None,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
    
    # Compilation
    logger.info('Compiling functions')
    corrector_trainner = Trainner(squared_error_sgd_mom(output_layer, lr=label_rate, mom=0, target_var=target_var), 
                                  'corrector',)
    corrector_trainner.preprocess = epoch_shuffle
    if hp_lambda != 0.0:
        domain_trainner = Trainner(adversarial([src_layer, output_layer], hp_lambda=hp_lambda,
                                              lr=domain_rate, mom=domain_mom),
                                   'domain')

    #=========================================================================
    # Train the Neural Network
    #=========================================================================
    if hp_lambda != 0.0:
        stats = training([corrector_trainner, domain_trainner], [corrector_data, domain_data],
                         num_epochs=num_epochs, logger=logger)
    else:
        stats = training([corrector_trainner,], [corrector_data,],
                     num_epochs=num_epochs, logger=logger)
    
    #=========================================================================
    # Print, Plot, Save the final results
    #=========================================================================
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
    
    # Plot the test data
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
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('fig/'+title+'-data.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
