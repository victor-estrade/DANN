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

from datasets.moon import load_moon
from datasets.utils import random_mat_dataset
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier
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


def epoch_shuffle(self):
    self['X_train'] = classwise_shuffle(self['X_train'], self['labels'])
    return self


def training(trainers, train_data, testers=[], test_data=[], num_epochs=20, logger=None):
    """
    TODO : Explain the whole function

    Params
    ------
        trainers:
        train_data:
        testers: (default=[])
        test_data: (default=[])
        num_epochs: (default=20)
        logger: (default=None)

    Return
    ------
        stats: dict with stats
    """
    if logger is None:
        logger = new_logger()

    logger.info("Starting training...")
    final_stats = {}
    final_stats.update({trainer.name+' training loss': [] for trainer in trainers})
    final_stats.update({trainer.name+' training acc': [] for trainer in trainers})
    final_stats.update({trainer.name+' valid loss': [] for trainer in trainers})
    final_stats.update({trainer.name+' valid acc': [] for trainer in trainers})
    final_stats.update({tester.name+' valid loss': [] for tester in testers})
    final_stats.update({tester.name+' valid acc': [] for tester in testers})

    
    for epoch in range(num_epochs):
        # Prepare the statistics
        start_time = time.time()
        stats = { key:[] for key in final_stats.keys()}

        # Do some trainning preparations :
        for data in train_data+test_data:
            if 'prepare' in data:
                data = data['prepare'](data)

        # Training : (forward and backward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_train'], data['y_train'], data['batchsize']) 
                        for data in train_data)
        for minibatches in zip(*batches):
            for batch, trainer in zip(minibatches, trainers):
                # X, y = batch
                loss, acc = trainer.train(*batch)
                stats[trainer.name+' training loss'].append(loss)
                stats[trainer.name+' training acc'].append(acc*100)
        
        # Validation (forward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_val'], data['y_val'], data['batchsize']) 
                        for data in train_data+test_data)
        for minibatches in zip(*batches):
            for batch, valider in zip(minibatches, trainers+testers):
                # X, y = batch
                loss, acc = valider.valid(*batch)
                stats[valider.name+' valid loss'].append(loss)
                stats[valider.name+' valid acc'].append(acc*100)
        
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        for stat_name, stat_value in sorted(stats.items()):
            if stat_value:
                mean_value = np.mean(stat_value)
                logger.info('   {:30} : {:.6f}'.format(
                    stat_name, mean_value))
                final_stats[stat_name].append(mean_value)

    return final_stats


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
        default=0.01, type=float, dest='domain_rate')
    parser.add_argument(
        '--domain-mom', help="The learning rate momentum of the domain part of the neural network ",
        default=0., type=float, dest='domain_mom')

    args = parser.parse_args()
    return args


def main():
    """
    The main function.
    """
    # Parse the aruments
    args = parseArgs()
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    label_rate = args.label_rate
    label_mom = args.label_mom
    domain_rate = args.domain_rate
    domain_mom = args.domain_mom

    # Set up the training :
    data_name = 'MoonRMat'
    batchsize = 32
    model = 'ClassWiseCorrector'
    title = '{}-{}-lambda-{:.4f}'.format(data_name, model, hp_lambda)

    # Load Moon Dataset
    source_data, target_data, domain_data = load_moon()
    source_data, target_data, domain_data = random_mat_dataset(source_data)
    
    corrector_data = dict(target_data)
    corrector_data.update({
    	'y_train':source_data['X_train'],
    	'y_val':source_data['X_val'],
    	'y_test':source_data['X_test'],
        'labels': source_data['y_train']
    	})
    corrector_data['prepare'] = epoch_shuffle

    # Prepare the logger :
    # f_log = log_fname(title)
    logger = new_logger()
    logger.info('Model: {}'.format(model))
    logger.info('Data: {}'.format(data_name))
    logger.info('hp_lambda = {:.4f}'.format(hp_lambda))

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    shape = (None, 2)
    input_layer = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    #=========================================================================
    # Build the neural network architecture
    #=========================================================================
    output_layer = lasagne.layers.DenseLayer(
                    input_layer,
                    num_units=np.prod(shape[1:]),
                    nonlinearity=None,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
    if hp_lambda != 0.0:
        rgl = ReverseGradientLayer(output_layer, hp_lambda=hp_lambda)
        domain_clf = Classifier(rgl, 2)
    
    # Compilation
    logger.info('Compiling functions')
    corrector_trainner = Trainner(output_layer, squared_error_sgd_mom(lr=label_rate, mom=0, target_var=target_var), 
    							 'corrector',)
    if hp_lambda != 0.0:
        domain_trainner = Trainner(domain_clf.output_layer, crossentropy_sgd_mom(lr=domain_rate, mom=domain_mom), 'domain')


    # Train the NN
    if hp_lambda != 0.0:
        stats = training([corrector_trainner, domain_trainner], [corrector_data, domain_data],
                         num_epochs=num_epochs, logger=logger)
    else:
        stats = training([corrector_trainner,], [corrector_data,],
                     num_epochs=num_epochs, logger=logger)
    
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
    fig.savefig('fig/'+title+'.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
