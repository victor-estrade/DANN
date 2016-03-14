#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import time
import gzip

import theano
import lasagne
import dann

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

from datasets import load_mnistM
from nn_compilers import compile_sgd
from logs import log_fname, new_logger
from utils import pop_last_line
from sklearn.datasets import make_moons as mkmoon


__author__ = 'Estrade Victor'


##############################################################################
##############################################################################
##############################################################################
##############################################################################
class Path(object):
    def __init__(self, nn, compiler, input_var=None, target_var=None, name='', batchsize=500):
        """Path class is a helper class to handle the training proscess.

        Params
        ------
            nn:
            compiler:
            name (default=''):
            batchsize(default=500):
        
        Example:
        --------
        >>> TODO

        """
        self.nn = nn
        self.compiler = compiler
        self.input_var = input_var
        self.target_var = target_var
        self.name = name
        self.batchsize = batchsize

        self.train_loss = []
        self.train_acc = []
        self.epoch = 0
        self.train_stats = {'loss':[], 'acc':[]}
        self.val_loss = []
        self.val_acc = []
        self.val_stats = {'loss':[], 'acc':[]}

    def compile(self):
        """Compile the neural network.

        Return
        ------
            self:
            input_var:
            target_var:
        """
        self.train_fn, self.test_fn, output_fn = self.compiler(self.nn, self.input_var, self.target_var)
        return self

    def train(self, X, y):
        """Do one training iteration over the given minibatch data.
        """
        # print('Path training batch shape:', X.shape, y.shape)
        # a = self.train_fn(X, y)
        # print('Path training a:', a)

        loss, acc = self.train_fn(X, y)
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def val(self, X, y):
        """Do one validation iteration over the given minibatch data.
        """
        loss, acc = self.test_fn(X, y)
        self.val_loss.append(loss)
        self.val_acc.append(acc)

    def end_epoch(self):
        """End a epoch, computes the statistics of this epoch
        """
        self.train_stats['loss'].append(np.mean(self.train_loss))
        self.train_loss = []
        self.train_stats['acc'].append(np.mean(self.train_acc))
        self.train_acc = []
        self.val_stats['loss'].append(np.mean(self.val_loss))
        self.val_loss = []
        self.val_stats['acc'].append(np.mean(self.val_acc))
        self.val_acc = []
        self.epoch = self.epoch+1


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Helper function interating over the given inputs

    Params
    ------
        inputs: the data (numpy array)
        targets: the target values (numpy array)
        batchsize: the batch size (int)
        shuffle (default=False):
    
    Return
    ------
        (input_slice, target_slice) as a generator
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def training(datasets, pathes, num_epochs=50, logger=None):
    """ training procedure. Used to train a multiple output network.
    """

    assert len(datasets) == len(pathes)

    if logger is None:
        logger = new_logger()

    # Compiling functions:
    logger.info("Compiling functions...")
    pathes = [path.compile() for path in pathes]

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        # training (forward and backward propagation)
        batches = tuple(iterate_minibatches(data['X_train'], data['y_train'], data['batchsize']) 
                        for data in datasets)
        for minibatches in zip(*batches):
            for batch, path in zip(minibatches, pathes):
                X, y = batch
                path.train(X, y)

        # Validation (forward propagation)
        batches = tuple(iterate_minibatches(data['X_val'], data['y_val'], data['batchsize']) for data in datasets)
        for minibatches in zip(*batches):
            for batch, path in zip(minibatches, pathes):
                X, y = batch
                path.val(X, y)
        # execute the ending code of the pathes
        [path.end_epoch() for path in pathes]

        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        for path in pathes:
            for stat_name, stat_value in path.train_stats.items():
                logger.info('   {:10} training {:10}: {:.6f}'.format(
                    path.name, stat_name, stat_value[-1]))
            for stat_name, stat_value in path.val_stats.items():
                logger.info('   {:10} valid    {:10}: {:.6f}'.format(
                    path.name, stat_name, stat_value[-1]))


def test(validation_fun, X, y, logger=None, batchsize=500):
    """
    Test the model using the given predict function on minibacth.
    Return the statistics.

    Params
    ------
        validation_fun: the predict function. Should take minibatch from X and y
            and return a loss value and an accuracy value :
            >>> loss, accuracy =  validation_fun(X, y)
        X: the input data
        y: the target value
        logger (default=None): used to output some information
        batchsize (default=500): the size on the minibatches

    Return
    ------
        stats: a dictionnary with 'loss' and 'acc'
    """
    # After training, we compute and print the test error:
    stats = {}
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X, y, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = validation_fun(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_err /= test_batches
    test_acc = (test_acc / test_batches) * 100
    if logger:
        logger.info("  {:30}: {:.6f}".format('source test loss',
            test_err))
        logger.info("  {:30}: {:.2f} %".format('source test accuracy',
            test_acc))

    # And saving them:
    stats['loss'] = test_err
    stats['acc'] = test_acc
    return stats


def rotate_data(X, angle=45.):
    """Apply a rotation on a 2D dataset.
    """
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    X_r = np.empty_like(X)
    X_r[:] = X[:].dot(rotMatrix)
    return X_r


if __name__ == '__main__':
    X, y = mkmoon(n_samples=5000, shuffle=True, noise=0.05, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X_r = rotate_data(X)

    X_train, X_val, X_test = X[0:3000], X[3000:4000], X[4000:]
    y_train, y_val, y_test = y[0:3000], y[3000:4000], y[4000:]
    
    X_r_train, X_r_val, X_r_test = X_r[0:3000], X_r[3000:4000], X_r[4000:]
    y_r_train, y_r_val, y_r_test = y[0:3000], y[3000:4000], y[4000:]
    
    data = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    data_r = ((X_r_train, y_r_train), (X_r_val, y_r_val), (X_r_test, y_r_test))
    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize':100,
                    }

    domain_data = {
                    'X_train': np.vstack([X_train, X_r_train]),
                    'y_train': np.hstack([np.zeros_like(y_train, dtype=np.int32), 
                               np.ones_like(y_r_train, dtype=np.int32)]),
                    'X_val': np.vstack([X_val, X_r_val]),
                    'y_val': np.hstack([np.zeros_like(y_val, dtype=np.int32), 
                               np.ones_like(y_r_val, dtype=np.int32)]),
                    'X_test': np.vstack([X_test, X_r_test]),
                    'y_test': np.hstack([np.zeros_like(y_test, dtype=np.int32), 
                               np.ones_like(y_r_test, dtype=np.int32)]),
                    'batchsize':200,
                    }
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    label_nn, domain_nn = dann.build_factory('small', input_var=input_var, 
                                             shape=(None, X_train.shape[1]))
    datas = [source_data, domain_data]
    pathes = [Path(label_nn, compile_sgd, input_var=input_var,
                    target_var=target_var, name='source'),
             Path(domain_nn, compile_sgd, input_var=input_var,
                    target_var=target_var, name='target'),
             ]

    training(datas, pathes, num_epochs=5)
