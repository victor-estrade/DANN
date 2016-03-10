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
from logs import log_fname, new_logger
from utils import pop_last_line


__author__ = 'Estrade Victor'


def main(model='cnn', num_epochs=500, hp_lambda=0.1, invert=False, logger=None):
    if logger is None:
        logger  = new_logger()
    # Load the dataset
    logger.info("Loading data...")
    source, target = load_mnistM(shape=(-1, 3, 28, 28))
    # Invert target and source
    if invert:
        source, target = target, source

    train_S, val_S, test_S = source
    train_T, val_T, test_T = target

    X_train_source, y_train = train_S
    X_train_target, _ = train_T

    X_val_source, y_val = val_S
    X_val_target, _ = val_T

    X_test_source, y_test = test_S
    X_test_target, _ = test_T

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model...")
    logger.info("hp_lambda : {}".format(hp_lambda))
    graphs = dann.build_factory(model, input_var=input_var)
    if type(graphs) is not tuple:
        label_predictor = graphs
        domain_predictor = None
    elif type(graphs) is tuple:
        label_predictor, domain_predictor = graphs
    else:
        logger.error("Unrecognized model type {} with code {}.".format(model))
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    label_prediction = lasagne.layers.get_output(label_predictor)
    label_loss = lasagne.objectives.categorical_crossentropy(label_prediction, target_var)
    label_loss = label_loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    label_params = lasagne.layers.get_all_params(label_predictor, trainable=True)
    # label_updates = lasagne.updates.nesterov_momentum(
    #         label_loss, label_params, learning_rate=0.01, momentum=0.9)
    label_updates = lasagne.updates.sgd(label_loss, label_params, learning_rate=0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(label_predictor, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    logger.info("Compiling functions...")
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], label_loss, updates=label_updates,
                               allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],
                             allow_input_downcast=True)

    if domain_predictor is not None:
        # Compile the domain aversial part of the network for trainning as 
        # adversial_fn
        logger.info("Compiling functions (domain adversial)...")

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        domain_prediction = lasagne.layers.get_output(domain_predictor)
        domain_loss = lasagne.objectives.categorical_crossentropy(domain_prediction, target_var)
        domain_loss = domain_loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD).
        domain_params = lasagne.layers.get_all_params(domain_predictor, trainable=True)
        # domain_updates = lasagne.updates.nesterov_momentum(
        #         domain_loss, domain_params, learning_rate=0.01, momentum=0.9)
        domain_updates = lasagne.updates.sgd(domain_loss, domain_params, learning_rate=0.01)
        domain_acc = T.mean(T.eq(T.argmax(domain_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        adversial_fn = theano.function([input_var, target_var], [domain_loss, domain_acc],
                                        updates=domain_updates,
                                        allow_input_downcast=True)

    # Finally, launch the training loop.
    logger.info("Starting training...")
    # Dictionnary saving the statistics
    stats = {}
    stats['source_val_acc'] = []
    stats['target_val_acc'] = []
    stats['source_val_loss'] = []
    stats['target_val_loss'] = []
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_domain_acc = 0
        train_domain_loss = 0
        train_batches = 0
        start_time = time.time()
        if domain_predictor is None:
            for batch in dann.iterate_minibatches(X_train_source, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
        else:
            for source_batch, target_batch in zip(dann.iterate_minibatches(X_train_source, y_train, 500, shuffle=True),
                            dann.iterate_minibatches(X_train_target, y_train, 500, shuffle=True)):
                X_source, y_source = source_batch
                train_err += train_fn(X_source, y_source)
                train_batches += 1
                X_target, y_target = target_batch
                X = np.vstack([X_source, X_target])
                y = np.hstack([np.zeros_like(y_source, dtype=np.int32), 
                               np.ones_like(y_target, dtype=np.int32)])
                train_domain_loss, train_domain_acc = adversial_fn(X, y)

        # And a full pass over the validation data:
        s = dann.test(val_fn, X_val_source, y_val)
        stats['source_val_loss'].append(s['loss'])
        stats['source_val_acc'].append(s['acc'])
        
        s = dann.test(val_fn, X_val_target, y_val)
        stats['target_val_loss'].append(s['loss'])
        stats['target_val_acc'].append(s['acc'])

        # Put the statistics and mesures in the log.
        # Trainning logs
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info("  {:30}: {:.6f}".format('training loss',
            train_err / train_batches))
        logger.info("  {:30}: {:.6f}".format('training domain loss',
            train_domain_loss / train_batches))
        logger.info("  {:30}: {:.2f} %".format('training domain acc',
            train_domain_acc / train_batches *100))
        # Validation logs
        logger.info("  {:30}: {:.6f}".format('source val loss',
            stats['source_val_loss'][-1]))
        logger.info("  {:30}: {:.2f} %".format('source val accuracy',
            stats['source_val_acc'][-1]))
        logger.info("  {:30}: {:.6f}".format('target val loss',
            stats['target_val_loss'][-1]))
        logger.info("  {:30}: {:.2f} %".format('target val accuracy',
            stats['target_val_loss'][-1]))

    # After training, we compute and print the test error:
    logger.info("Final results:")
    s = dann.test(val_fn, X_test_source, y_test)
    stats['source_test_loss'] = s['loss']
    stats['source_test_acc'] = s['acc']
    
    s = dann.test(val_fn, X_test_target, y_test)
    stats['target_test_loss'] = s['loss']
    stats['target_test_acc'] = s['acc']

    # Test logs
    logger.info("  {:30}: {:.6f}".format('source test loss',
        stats['source_test_loss']))
    logger.info("  {:30}: {:.2f} %".format('source test accuracy',
        stats['source_test_acc']))
    logger.info("  {:30}: {:.6f}".format('target test loss',
        stats['target_test_loss']))
    logger.info("  {:30}: {:.2f} %".format('target test accuracy',
        stats['target_test_loss']))
        
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return stats


if __name__ == '__main__':
    np.savetxt('TODO.txt', np.logspace(-1, 0, num=5))
    
    todo = pop_last_line('TODO.txt')
    while todo:
        hp_lambda = float(todo)
        model = 'small'
        title = 'S-T-{}-lambda-{:.3f}-'.format(model, hp_lambda)
        for i in range(3):
            f_log = log_fname(title)
            logger = new_logger(f_log)
            
            stats = main(model=model, num_epochs=4, hp_lambda=hp_lambda, 
                         invert=False, logger=logger)

            plt.plot(stats['source_val_acc'], label='source', c='blue')
            plt.plot(stats['target_val_acc'], label='target', c='red')
            plt.axhline(y=stats['source_test_acc'], c='blue')
            plt.axhline(y=stats['target_test_acc'], c='green')
        plt.xlabel('epoch')
        plt.ylabel('accuracy (%)')
        plt.title(title)
        plt.legend()
        plt.savefig('fig/'+title+'.png')
        plt.clf() # Clear plot window
        todo = pop_last_line('TODO.txt')

