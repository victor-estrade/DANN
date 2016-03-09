#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import time
import gzip

import theano
import lasagne

import cPickle as pickle
import numpy as np
import theano.tensor as T
# import pandas as pd
import matplotlib.pyplot as plt

from datasets import mnist_blend
from logs import log_fname, new_logger
"""
http://stackoverflow.com/questions/33879736/can-i-selectively-invert-theano-gradients-during-backpropagation
"""


class ReverseGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


class ReverseGradientLayer(lasagne.layers.Layer):
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.op = ReverseGradient(hp_lambda)

    def get_output_for(self, input, **kwargs):
        return self.op(input)


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    cv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    cv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(feature, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return network


def build_dann(input_var=None, hp_lambda=0.5):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    cv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 48 5x5 kernels, and another 2x2 pooling:
    cv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    label_predictor = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(feature, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    domain_predictor = lasagne.layers.DenseLayer(
            ReverseGradientLayer(feature, hp_lambda=hp_lambda),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor


def build_small_dann(input_var=None, hp_lambda=0.5):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    # cv1 = lasagne.layers.Conv2DLayer(
    #         input_layer, num_filters=32, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         # W=lasagne.init.GlorotUniform(),
    #         )
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # pool1 = lasagne.layers.MaxPool2DLayer(cv1, pool_size=(2, 2))

    # Another convolution with 48 5x5 kernels, and another 2x2 pooling:
    # cv2 = lasagne.layers.Conv2DLayer(
    #         pool1, num_filters=48, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         # W=lasagne.init.GlorotUniform(),
    #         )
    # pool2 = lasagne.layers.MaxPool2DLayer(cv2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    feature = lasagne.layers.DenseLayer(
            input_layer,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    label_predictor = lasagne.layers.DenseLayer(
            feature,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    # Domain classifier
    domain_hidden = lasagne.layers.DenseLayer(
            ReverseGradientLayer(feature, hp_lambda=hp_lambda),
            num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify,
            # W=lasagne.init.GlorotUniform(),
            )
    domain_predictor = lasagne.layers.DenseLayer(
            domain_hidden,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax,
            # W=lasagne.init.GlorotUniform(),
            )

    return label_predictor, domain_predictor



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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


def main(model='cnn', num_epochs=500, hp_lambda=0.1, invert=False, logger=None):
    if logger is None:
        logger  = new_logger()
    # Load the dataset
    logger.info("Loading data...")
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_S, valid_S, test_S = pickle.load(f)

    X_train_source, y_train = train_S
    X_train_source, X_train_target = mnist_blend(X_train_source)
    X_train_source = X_train_source.reshape(-1, 3, 28, 28)
    X_train_target = X_train_target.reshape(-1, 3, 28, 28)

    X_val_source, y_val = valid_S
    X_val_source, X_val_target = mnist_blend(X_val_source)
    X_val_source = X_val_source.reshape(-1, 3, 28, 28)
    X_val_target = X_val_target.reshape(-1, 3, 28, 28)

    X_test_source, y_test = test_S
    X_test_source, X_test_target = mnist_blend(X_test_source)
    X_test_source = X_test_source.reshape(-1, 3, 28, 28)
    X_test_target = X_test_target.reshape(-1, 3, 28, 28)

    # Invert target and source
    if invert:
        X_train_source, X_train_target = X_train_target, X_train_source
        X_val_source, X_val_target = X_val_target, X_val_source
        X_test_source, X_test_target = X_test_target, X_test_source


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model...")
    if model == 'dann':
        label_predictor, domain_predictor = build_dann(input_var, hp_lambda=hp_lambda)
        logger.info('hp_lambda: {}'.format(hp_lambda))
    elif model == 'cnn':
        label_predictor = build_cnn(input_var)
        domain_predictor = None
    elif model == 'small':
        label_predictor, domain_predictor = build_small_dann(input_var, hp_lambda=hp_lambda)
        logger.info('hp_lambda: {}'.format(hp_lambda))
    else:
        logger.error("Unrecognized model type {}.".format(model))
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
    label_updates = lasagne.updates.nesterov_momentum(
            label_loss, label_params, learning_rate=0.01, momentum=0.9)

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
            for batch in iterate_minibatches(X_train_source, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
        else:
            for source_batch, target_batch in zip(iterate_minibatches(X_train_source, y_train, 500, shuffle=True),
                            iterate_minibatches(X_train_target, y_train, 500, shuffle=True)):
                X_source, y_source = source_batch
                train_err += train_fn(X_source, y_source)
                train_batches += 1
                X_target, y_target = target_batch
                X = np.vstack([X_source, X_target])
                y = np.hstack([np.zeros_like(y_source, dtype=np.int32), 
                               np.ones_like(y_target, dtype=np.int32)])
                train_domain_loss, train_domain_acc = adversial_fn(X, y)

        # And a full pass over the validation data:
        val_err_source = 0
        val_acc_source = 0
        val_batches_source = 0
        for batch in iterate_minibatches(X_val_source, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err_source += err
            val_acc_source += acc
            val_batches_source += 1
        
        val_err_target = 0
        val_acc_target = 0
        val_batches_target = 0
        for batch in iterate_minibatches(X_val_target, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err_target += err
            val_acc_target += acc
            val_batches_target += 1

        # Then we print the results for this epoch:
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info("  {:30}: {:.6f}".format('training loss',
            train_err / train_batches))
        logger.info("  {:30}: {:.6f}".format('training domain loss',
            train_domain_loss / train_batches))
        logger.info("  {:30}: {:.2f} %".format('training domain acc',
            train_domain_acc / train_batches *100))
        logger.info("  {:30}: {:.6f}".format('source validation loss',
            val_err_source / val_batches_source))
        logger.info("  {:30}: {:.2f} %".format('source validation accuracy',
            val_acc_source / val_batches_source * 100))
        logger.info("  {:30}: {:.6f}".format('target validation loss',
            val_err_target / val_batches_target))
        logger.info("  {:30}: {:.2f} %".format('target validation accuracy',
            val_acc_target / val_batches_target * 100))
        # And saving them:
        stats['source_val_loss'].append(val_err_source / val_batches_source)
        stats['source_val_acc'].append(val_acc_source / val_batches_source * 100)
        stats['target_val_loss'].append(val_err_target / val_batches_target)
        stats['target_val_acc'].append(val_acc_target / val_batches_target * 100)

    # After training, we compute and print the test error:
    test_err_source = 0
    test_acc_source = 0
    test_batches_source = 0
    for batch in iterate_minibatches(X_test_source, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err_source += err
        test_acc_source += acc
        test_batches_source += 1

    test_err_target = 0
    test_acc_target = 0
    test_batches_target = 0
    for batch in iterate_minibatches(X_test_target, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err_target += err
        test_acc_target += acc
        test_batches_target += 1
        
    logger.info("Final results:")
    logger.info("  {:30}: {:.6f}".format('source test loss',
        test_err_source / test_batches_source))
    logger.info("  {:30}: {:.2f} %".format('source test accuracy',
        test_acc_source / test_batches_source * 100))
    logger.info("  {:30}: {:.6f}".format('target test loss',
        test_err_target / test_batches_target))
    logger.info("  {:30}: {:.2f} %".format('target test accuracy',
        test_acc_target / test_batches_target * 100))

    # And saving them:
    stats['source_test_loss'] = val_err_source / val_batches_source
    stats['source_test_acc'] = val_acc_source / val_batches_source * 100
    stats['target_test_loss'] = val_err_target / val_batches_target
    stats['target_test_acc'] = val_acc_target / val_batches_target * 100
        
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return stats

if __name__ == '__main__':
    
    hp_lambda = 0.2
    model = 'small'
    stats = main(model=model, num_epochs=50, hp_lambda=hp_lambda, 
                 invert=False)

    plt.plot(stats['source_val_acc'], label='source')
    plt.plot(stats['target_val_acc'], label='target')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    # plt.plot(stats['source_loss'])
    # plt.plot(stats['target_loss'])
    plt.legend()
    title = 'S-T-{}-lambda-{:3f}'.format(model, hp_lambda)
    plt.savefig('fig/'+title+'.png')
