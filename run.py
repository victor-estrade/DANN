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

from datasets import load_mnistM, load_moon
from nn_compilers import compile_sgd, compile_nesterov
from logs import log_fname, new_logger
from utils import pop_last_line
from sklearn.datasets import make_moons as mkmoon



##############################################################################
##############################################################################
##############################################################################
##############################################################################
class Path(object):
    def __init__(self, nn, compiler, input_var=None, target_var=None, name='',
                batchsize=500, trainable=True):
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
        self.trainable = trainable
        self.batchsize = batchsize

        self.train_loss = []
        self.train_acc = []
        self.epoch = 0
        if trainable:
            self.train_stats = {'loss':[], 'acc':[]}
        else:
            self.train_stats = {}
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
        self.train_fn, self.test_fn, self.output_fn = self.compiler(self.nn, self.input_var, self.target_var)
        return self

    def train(self, X, y, save=True):
        """Do one training iteration over the given minibatch data.
        """
        # print('Path training batch shape:', X.shape, y.shape)
        # a = self.train_fn(X, y)
        # print('Path training a:', a)
        if self.trainable:
            loss, acc = self.train_fn(X, y)
            if save:
                self.train_loss.append(loss)
                self.train_acc.append(acc)

    def val(self, X, y, save=True):
        """Do one validation iteration over the given minibatch data.
        """
        loss, acc = self.test_fn(X, y)
        if save:
            self.val_loss.append(loss)
            self.val_acc.append(acc)

    def end_epoch(self):
        """End a epoch, computes the statistics of this epoch
        """
        if self.trainable:
            self.train_stats['loss'].append(np.mean(self.train_loss))
            self.train_loss = []
            self.train_stats['acc'].append(np.mean(self.train_acc)*100)
            self.train_acc = []
        self.val_stats['loss'].append(np.mean(self.val_loss))
        self.val_loss = []
        self.val_stats['acc'].append(np.mean(self.val_acc)*100)
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


def test(path, X, y, logger=None, batchsize=500):
    """
    Test the model using the given predict function on minibacth.
    Return the statistics.

    Params
    ------
        path: the predict function. Should take minibatch from X and y
            and return a loss value and an accuracy value :
            >>> loss, accuracy =  path.test_fn(X, y)
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
        err, acc = path.test_fn(inputs, targets, save=False)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_err /= test_batches
    test_acc = (test_acc / test_batches) * 100
    if logger:
        logger.info('   {:10} testing {:10}: {:.6f}'.format(
                path.name, stat_name, test_err))
        logger.info('   {:10} testing {:10}: {:.6f}'.format(
                path.name, stat_name, test_acc))
        
    # And saving them:
    stats['loss'] = test_err
    stats['acc'] = test_acc
    return stats


def plot_bound(X, y, predict_fn):
    from matplotlib.colors import ListedColormap
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)[0, :, 1]
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


if __name__ == '__main__':
    # Moon Dataset
    data_name = 'moon'
    batchsize = 32
    source_data, target_data, domain_data = load_moon()
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    shape = (None, X_train.shape[1])


    # MNIST dataset
    data_name = 'MNIST'
    batchsize = 500
    source_data, target_data, domain_data = load_mnistM(shape=(-1, 3, 28, 28))
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    shape = (None, 3, 28, 28)


    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize':batchsize,
                    }

    target_data = {
                    'X_train': X_t_train,
                    'y_train': y_t_train,
                    'X_val': X_t_val,
                    'y_val': y_t_val,
                    'X_test': X_t_test,
                    'y_test': y_t_test,
                    'batchsize':batchsize,
                    }

    domain_data = {
                    'X_train': np.vstack([X_train, X_t_train]),
                    'y_train': np.hstack([np.zeros_like(y_train, dtype=np.int32), 
                               np.ones_like(y_t_train, dtype=np.int32)]),
                    'X_val': np.vstack([X_val, X_t_val]),
                    'y_val': np.hstack([np.zeros_like(y_val, dtype=np.int32), 
                               np.ones_like(y_t_val, dtype=np.int32)]),
                    'X_test': np.vstack([X_test, X_t_test]),
                    'y_test': np.hstack([np.zeros_like(y_test, dtype=np.int32), 
                               np.ones_like(y_t_test, dtype=np.int32)]),
                    'batchsize':batchsize*2,
                    }

    # Gather the data in the right order
    datas = [source_data, 
            # domain_data, 
            # target_data,
            ]

    model = 'cnn'
    hp_lambda = 0.

    fig, ax = plt.subplots()
    # for i in range(1):
    for hp_lambda in [0.,]:#+list(np.logspace(-1, 1, num=10)):
        title = '{}-lambda-{:.4f}-{}'.format(model, hp_lambda, data_name)
        f_log = log_fname(title)
        logger = new_logger(f_log)
        logger.info('hp_lambda = {:.4f}'.format(hp_lambda))
        
        # Build the neural network architecture
        # label_nn, domain_nn = dann.build_factory(model, input_var=input_var, 
        #                                          shape=shape,
        #                                          hp_lambda=hp_lambda)
        label_nn = dann.build_factory(model, input_var=input_var, 
                                             shape=shape,
                                             hp_lambda=hp_lambda)
        # Build the pathes to prepare the training
        label_path = Path(label_nn, compile_nesterov, input_var=input_var,
                        target_var=target_var, name='source')
        # domain_path = Path(domain_nn, compile_nesterov, input_var=input_var,
        #                 target_var=target_var, name='domain')
        # target_path = Path(label_nn, compile_sgd, input_var=input_var,
        #                 target_var=target_var, name='target', trainable=False)
        pathes = [label_path,
                  # domain_path,
                  # target_path,
                  ]

        # Train the NN
        training(datas, pathes, num_epochs=350)
        
        # Plot learning accuracy curve
        fig0, ax0 = plt.subplots()
        ax0.plot(label_path.val_stats['acc'], label='source', c='blue')
        ax0.plot(target_path.val_stats['acc'], label='target', c='red')
        # ax0.axhline(y=stats['source_test_acc'], c='blue')
        # ax0.axhline(y=stats['target_test_acc'], c='green')
        ax0.set_xlabel('epoch')
        ax0.set_ylabel('accuracy')
        ax0.set_ylim(0., 100.0)
        ax0.set_title(title)
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig0.savefig('fig/'+title+'.png', bbox_inches='tight')
        fig0.clf() # Clear plot window
        
        # Plot evolution
        ax.plot(target_path.val_stats['acc'], label='l={}'.format(hp_lambda))
        ax.set_title('Evolution')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('fig/Evolution'+title+'.png', bbox_inches='tight')
        

    if data_name == 'moon':
        colors = 'rb'
        plt.scatter(X[:, 0], X[:, 1], c=[colors[l] for l in y])
        plt.title('Moon dataset')
        plt.savefig('fig/moon.png')
        plt.clf() # Clear plot window
        
        plt.scatter(X_r[:, 0], X_r[:, 1], c=[colors[l] for l in y])
        plt.title('Moon rotated dataset')
        plt.savefig('fig/moon-rotated.png')
        plt.clf() # Clear plot window
        
        plot_bound(X, y, label_path.output_fn)
        plt.title('Moon bounds')
        plt.savefig('fig/moon-bound.png')
        plt.clf() # Clear plot window
        
        plot_bound(X_r, y, target_path.output_fn)
        plt.title('Moon rot bounds')
        plt.savefig('fig/moon-rot-bound.png')
        plt.clf() # Clear plot window
