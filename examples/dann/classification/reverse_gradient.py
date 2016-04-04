#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import theano
import theano.tensor as T
import lasagne

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from datasets.moon import load_moon
from logs import log_fname, new_logger
from nn.dann import AbstractDANN
from nn.rgl import ReverseGradientLayer
from nn.compilers import crossentropy_sgd_mom
from utils import iterate_minibatches, plot_bound


class BadNN(AbstractDANN):
    """
    A NN with a reversal gradient layer that will destroy the performances.
    """

    def __init__(self, nb_units, nb_output, input_layer, hp_lambda=-1):
        
        self.nb_output = nb_output
        self.hp_lambda = hp_lambda
        self.input_layer = input_layer
        self.nb_units = nb_units
        self.target_var = T.ivector('targets')
        self._build()

    def _build(self) :
        """
        Build the architecture of the neural network
        """
        feature = self.input_layer
        feature = lasagne.layers.DenseLayer(
                feature,
                num_units=self.nb_units,
                nonlinearity=lasagne.nonlinearities.tanh,
                # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                )
        self.feature = feature
        # Reversal gradient layer
        self.RGL = ReverseGradientLayer(self.feature, hp_lambda=self.hp_lambda)
        
        # Label classifier
        self.label_predictor = lasagne.layers.DenseLayer(
                self.RGL,
                num_units=self.nb_output,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.GlorotUniform(),
                )

    def training(self, source, num_epochs=50, logger=None):
        """ training procedure. Used to train a multiple output network.
        """

        if logger is None:
            logger = new_logger()

        logger.info("Starting training...")
        final_stats = {
                'source training loss': [], 'source training acc': [],
                'source valid loss': [], 'source valid acc': [],
                }

        for epoch in range(num_epochs):
            start_time = time.time()
            stats = { key:[] for key in final_stats.keys()}
            # training (forward and backward propagation)
            source_batches = iterate_minibatches(source['X_train'], source['y_train'], source['batchsize'], shuffle=True)
            for source_batch in source_batches:
                X, y = source_batch
                loss, acc = self.train_label(X, y)
                stats['source training loss'].append(loss)
                stats['source training acc'].append(acc*100)
                
            # Validation (forward propagation)
            source_batches = iterate_minibatches(source['X_val'], source['y_val'], source['batchsize'])
            for source_batch in source_batches:
                X, y = source_batch
                loss, acc = self.valid_label(X, y)
                stats['source valid loss'].append(loss)
                stats['source valid acc'].append(acc*100)

            logger.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            for stat_name, stat_value in sorted(stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    logger.info('   {:30} : {:.6f}'.format(
                        stat_name, mean_value))
                    final_stats[stat_name].append(mean_value)

        return final_stats


def main(hp_lambda=0.0, num_epochs=50, angle=-35, label_rate=1):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'MoonRotated'
    batchsize = 32
    source_data, target_data, domain_data = load_moon(angle=angle)

    # Set up the training :
    datas = [source_data, domain_data, target_data]

    model = 'BadNN'

    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)
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
    dann = BadNN(3, 2, input_layer, hp_lambda=hp_lambda)

    logger.info('Compiling functions')
    dann.compile_label(crossentropy_sgd_mom(lr=label_rate, mom=0))
    
    # Train the NN
    stats = dann.training(source_data, num_epochs=num_epochs)

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
    plt.close(fig)  # Clear plot window

    # Plot boundary :
    X = np.vstack([source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    plot_bound(X, y, dann.proba_label)
    plt.title('Moon bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window
    plt.close('all')


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
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    angle = args.angle
    label_rate = args.label_rate
    main(hp_lambda=hp_lambda,  num_epochs=num_epochs, angle=angle,
        label_rate=label_rate,)
