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

from datasets.mnist import load_mnist_mirror
from logs import log_fname, new_logger
from nn.dann import ShallowDANN, DenseDANN
from nn.compilers import compiler_sgd_mom
from utils import plot_bound, save_confusion_matrix
from sklearn.metrics import confusion_matrix


def main(hp_lambda=0.0, num_epochs=50, label_rate=1, domain_rate=1):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'MirrorMnist'
    batchsize = 500
    source_data, target_data, domain_data = load_mnist_mirror()

    # Set up the training :
    datas = [source_data, domain_data, target_data]

    model = 'DenseDANN[250-50]'

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
    dann = DenseDANN([250, 50], 10, input_layer, hp_lambda=hp_lambda)

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
    plt.close(fig) # Clear plot window
    
    # Sample image:
    fig = plt.figure()
    n_sample = 4
    rand = np.random.RandomState()
    for n in range(n_sample):
        i = rand.randint(source_data['X_test'].shape[0])
        sample_src = source_data['X_test'][i]
        sample_trg = target_data['X_test'][i]
        ax = fig.add_subplot(n_sample, 2, n*2+1)
        ax.axis('off')
        ax.imshow(sample_src, cmap='Greys_r')
        label = dann.predict_label(sample_src[np.newaxis])[0]
        ax.set_title('Source image (pred={})'.format(label))
        ax = fig.add_subplot(n_sample, 2, n*2+2)
        ax.axis('off')
        ax.imshow(sample_trg, cmap='Greys_r')
        label = dann.predict_label(sample_trg[np.newaxis])[0]
        ax.set_title('Target image (pred={})'.format(label))
    fig.savefig('fig/MNIST-sample.png')
    plt.close(fig) # Clear plot window
    
    # Plot confusion matrices :
    # Plot Target Test confusion matrix :
    X, y = target_data['X_test'], target_data['y_test']
    y_pred = np.asarray(dann.predict_label(X)).reshape(-1)
    save_confusion_matrix(y, y_pred, title=title+'TARGET-CM')
    
    # Plot Target Test confusion matrix :
    X, y = source_data['X_test'], source_data['y_test']
    y_pred = np.asarray(dann.predict_label(X)).reshape(-1)
    save_confusion_matrix(y, y_pred, title=title+'SOURCE-CM')

    # Plot Domain Test confusion matrix :
    X, y = domain_data['X_test'], domain_data['y_test']
    y_pred = np.asarray(dann.predict_domain(X)).reshape(-1)
    save_confusion_matrix(y, y_pred, title=title+'DOMAIN-CM')
    

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
