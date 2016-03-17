#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import time

import theano
import lasagne

import numpy as np
import theano.tensor as T

from rgl import ReverseGradientLayer
from logs import log_fname, new_logger
from utils import iterate_minibatches

class ShallowDANN(object):
    """
    A shallow DANN
    """

    def __init__(self, nb_units, nb_output, input_layer, nb_domain=2, hp_lambda=0):
        
        self.nb_output = nb_output
        self.nb_domain = nb_domain
        self.hp_lambda = hp_lambda
        self.input_layer = input_layer
        self.nb_units = nb_units
        self.target_var = T.ivector('targets')
        self._build()

    def _build(self) :
        """
        Build the architecture of the neural network
        """
        self.feature = lasagne.layers.DenseLayer(
                self.input_layer,
                num_units=self.nb_units,
                nonlinearity=lasagne.nonlinearities.tanh,
                # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                )

        # Reversal gradient layer
        self.RGL = ReverseGradientLayer(self.feature, hp_lambda=self.hp_lambda)
        
        # Label classifier
        self.label_predictor = lasagne.layers.DenseLayer(
                self.feature,
                num_units=self.nb_output,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.GlorotUniform(),
                )
        # Domain predictor
        self.domain_predictor = lasagne.layers.DenseLayer(
                self.RGL,
                # domain_hidden,
                num_units=self.nb_domain,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.GlorotUniform(),
                )

    def compile_label(self, compiler):
        self.train_label, self.predict_label, self.valid_label, self.proba_label = compiler(self.label_predictor)

    def compile_domain(self, compiler):
        self.train_domain, self.predict_domain, self.valid_domain, self.proba_domain = compiler(self.domain_predictor)

    def training(self, source, domain, target=None, num_epochs=50, logger=None):
        """ training procedure. Used to train a multiple output network.
        """

        if logger is None:
            logger = new_logger()

        logger.info("Starting training...")

        for epoch in range(num_epochs):
            start_time = time.time()
            stats = {
                    'domain training loss': [], 'domain training acc': [],
                    'domain valid loss': [], 'domain valid acc': [],
                    'source training loss': [], 'source training acc': [],
                    'source valid loss': [], 'source valid acc': [],
                    'target training loss': [], 'target training acc': [],
                    'target valid loss': [], 'target valid acc': [],
                    }
            # training (forward and backward propagation)
            source_batches = iterate_minibatches(source['X_train'], source['y_train'], source['batchsize'], shuffle=True)
            domain_batches = iterate_minibatches(domain['X_train'], domain['y_train'], domain['batchsize'], shuffle=True)
            for source_batch, domain_batch in zip(*(source_batches, domain_batches)):
                X, y = source_batch
                loss, acc = self.train_label(X, y)
                stats['source training loss'].append(loss)
                stats['source training acc'].append(acc*100)
                X, y = domain_batch
                loss, acc = self.train_domain(X, y)
                stats['domain training loss'].append(loss)
                stats['domain training acc'].append(acc*100)

            # Validation (forward propagation)
            source_batches = iterate_minibatches(source['X_val'], source['y_val'], source['batchsize'])
            domain_batches = iterate_minibatches(domain['X_val'], domain['y_val'], domain['batchsize'])
            for source_batch, domain_batch in zip(*(source_batches, domain_batches)):
                X, y = source_batch
                loss, acc = self.valid_label(X, y)
                stats['source valid loss'].append(loss)
                stats['source valid acc'].append(acc*100)
                X, y = domain_batch
                loss, acc = self.valid_domain(X, y)
                stats['domain valid loss'].append(loss)
                stats['domain valid acc'].append(acc*100)

            if target is not None:
                target_batches = iterate_minibatches(target['X_val'], target['y_val'], target['batchsize'])
                for target_batch in target_batches:
                    X, y = source_batch
                    loss, acc = self.valid_label(X, y)
                    stats['target valid loss'].append(loss)
                    stats['target valid acc'].append(acc*100)

            logger.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            for stat_name, stat_value in sorted(stats.items()):
                if stat_value:
                    logger.info('   {:30} : {:.6f}'.format(
                        stat_name, np.mean(stat_value)))


class DenseDANN(object):
    """
    A shallow DANN
    """

    def __init__(self, arch, nb_output, input_layer, nb_domain=2, hp_lambda=0):
        
        self.nb_output = nb_output
        self.nb_domain = nb_domain
        self.hp_lambda = hp_lambda
        self.input_layer = input_layer
        self.arch = arch
        self.target_var = T.ivector('targets')
        self._build()

    def _build(self) :
        """
        Build the architecture of the neural network
        """
        feature = self.input_layer
        for nb_units in arch:
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
                self.feature,
                num_units=self.nb_output,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.GlorotUniform(),
                )
        # Domain predictor
        self.domain_predictor = lasagne.layers.DenseLayer(
                self.RGL,
                # domain_hidden,
                num_units=self.nb_domain,
                nonlinearity=lasagne.nonlinearities.softmax,
                # W=lasagne.init.GlorotUniform(),
                )

    def compile_label(self, compiler):
        self.train_label, self.predict_label, self.valid_label, self.proba_label = compiler(self.label_predictor)

    def compile_domain(self, compiler):
        self.train_domain, self.predict_domain, self.valid_domain, self.proba_domain = compiler(self.domain_predictor)

    def training(self, source, domain, target=None, num_epochs=50, logger=None):
        """ training procedure. Used to train a multiple output network.
        """

        if logger is None:
            logger = new_logger()

        logger.info("Starting training...")

        for epoch in range(num_epochs):
            start_time = time.time()
            stats = {
                    'domain training loss': [], 'domain training acc': [],
                    'domain valid loss': [], 'domain valid acc': [],
                    'source training loss': [], 'source training acc': [],
                    'source valid loss': [], 'source valid acc': [],
                    'target training loss': [], 'target training acc': [],
                    'target valid loss': [], 'target valid acc': [],
                    }
            # training (forward and backward propagation)
            source_batches = iterate_minibatches(source['X_train'], source['y_train'], source['batchsize'], shuffle=True)
            domain_batches = iterate_minibatches(domain['X_train'], domain['y_train'], domain['batchsize'], shuffle=True)
            for source_batch, domain_batch in zip(*(source_batches, domain_batches)):
                X, y = source_batch
                loss, acc = self.train_label(X, y)
                stats['source training loss'].append(loss)
                stats['source training acc'].append(acc*100)
                X, y = domain_batch
                loss, acc = self.train_domain(X, y)
                stats['domain training loss'].append(loss)
                stats['domain training acc'].append(acc*100)

            # Validation (forward propagation)
            source_batches = iterate_minibatches(source['X_val'], source['y_val'], source['batchsize'])
            domain_batches = iterate_minibatches(domain['X_val'], domain['y_val'], domain['batchsize'])
            for source_batch, domain_batch in zip(*(source_batches, domain_batches)):
                X, y = source_batch
                loss, acc = self.valid_label(X, y)
                stats['source valid loss'].append(loss)
                stats['source valid acc'].append(acc*100)
                X, y = domain_batch
                loss, acc = self.valid_domain(X, y)
                stats['domain valid loss'].append(loss)
                stats['domain valid acc'].append(acc*100)

            if target is not None:
                target_batches = iterate_minibatches(target['X_val'], target['y_val'], target['batchsize'])
                for target_batch in target_batches:
                    X, y = source_batch
                    loss, acc = self.valid_label(X, y)
                    stats['target valid loss'].append(loss)
                    stats['target valid acc'].append(acc*100)

            logger.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            for stat_name, stat_value in sorted(stats.items()):
                if stat_value:
                    logger.info('   {:30} : {:.6f}'.format(
                        stat_name, np.mean(stat_value)))


if __name__ == '__main__':
    print("Hello !")
