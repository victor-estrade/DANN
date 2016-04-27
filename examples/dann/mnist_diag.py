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

from ..datasets.mnist import load_mnist
from ..datasets.transform import diag_dataset
from ..datasets.utils import make_domain_dataset
from ..logs import log_fname, new_logger
from ..nn.clone import clone_layer
from ..nn.compilers import crossentropy_sgd_mom, adversarial
from ..nn.training import Trainner, training

from utils import plot_bound, save_confusion_matrix
from sklearn.metrics import confusion_matrix


def parseArgs():
    """
    ArgumentParser.

    Return
    ------
        args: the parsed arguments.
    """
    # Retrieve the arguments
    parser = argparse.ArgumentParser(
        description="MNIST Diag adaptation example")
    parser.add_argument(
        '--epoch', help='Number of epoch in the training session',
        default=50, type=int, dest='num_epochs')
    parser.add_argument(
        '--batchsize', help='The mini-batch size',
        default=500, type=int, dest='batchsize')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0.01, type=float, dest='hp_lambda')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=1, type=float, dest='label_rate')
    parser.add_argument(
        '--label-mom', help="The learning rate momentum of the label part of the neural network ",
        default=0., type=float, dest='label_mom')
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
    num_epochs = args.num_epochs
    batchsize = args.batchsize
    hp_lambda = args.hp_lambda
    label_rate = args.label_rate
    label_mom = args.label_mom
    domain_rate = args.domain_rate
    domain_mom = args.domain_mom

    # Set up the naming information :
    data_name = 'MnistDiag'
    model = 'SimplestDANN'
    title = '{}-{}-lambda-{:.2e}'.format(data_name, model, hp_lambda)

    #=========================================================================
    # Load, Generate the datasets
    #=========================================================================
    # Load MNIST Dataset
    source_data = load_mnist(batchsize=batchsize)
    target_data = diag_dataset(source_data)
    domain_data = make_domain_dataset([source_data, target_data])

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
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')
    shape = (batchsize, 28, 28)

    # Build the layers
    input_layer = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    feature = lasagne.layers.DenseLayer(
                input_layer,
                num_units=50,
                nonlinearity=lasagne.nonlinearities.tanh,
                )
    label_layer = lasagne.layers.DenseLayer(
                feature,
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax,
                )
    
    input_layer2 = lasagne.layers.InputLayer(shape=shape, input_var=T.tensor3('inputs2'))
    feature2 = clone_layer(feature, input_layer2)
    #label_layer2 = clone_layer(label_layer, feature)
    
    # Compilation
    logger.info('Compiling functions')
    label_trainner = Trainner(crossentropy_sgd_mom(label_layer, lr=label_rate, mom=label_mom), 'source')
    domain_trainner = Trainner(adversarial([feature, feature2], hp_lambda=hp_lambda, lr=domain_rate, mom=domain_mom),'domain')
    target_tester = Trainner(crossentropy_sgd_mom(label_layer, lr=label_rate, mom=label_mom), 'target')

    #=========================================================================
    # Train the Neural Network
    #=========================================================================
    stats = training([label_trainner, domain_trainner], [source_data, domain_data],
                     testers=[target_tester,], test_data=[target_data],
                     num_epochs=num_epochs, logger=logger)
    
    #=========================================================================
    # Print, Plot, Save the final results
    #=========================================================================
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
        label = label_trainner.predict(sample_src[np.newaxis])[0]
        ax.set_title('Source image (pred={})'.format(label))
        ax = fig.add_subplot(n_sample, 2, n*2+2)
        ax.axis('off')
        ax.imshow(sample_trg, cmap='Greys_r')
        label = label_trainner.predict(sample_trg[np.newaxis])[0]
        ax.set_title('Target image (pred={})'.format(label))
    fig.savefig('fig/{}-sample.png'.format(title))
    plt.close(fig) # Clear plot window

    # Plot confusion matrices :
    # Plot Target Test confusion matrix :
    X, y = target_data['X_test'], target_data['y_test']
    y_pred = np.asarray(label_trainner.predict(X)).reshape(-1)
    save_confusion_matrix(y, y_pred, title=title+'TARGET-CM')
    
    # Plot Target Test confusion matrix :
    X, y = source_data['X_test'], source_data['y_test']
    y_pred = np.asarray(label_trainner.predict(X)).reshape(-1)
    save_confusion_matrix(y, y_pred, title=title+'SOURCE-CM')

    # Plot Domain Test confusion matrix :
    X = domain_data['X_test']
    y_pred = np.asarray(domain_trainner.predict(*X)).reshape(-1)
    y = np.hstack([np.ones(arr.shape[0], dtype=np.int32)*i 
                        for i, arr in enumerate(X)])
    save_confusion_matrix(y, y_pred, title=title+'DOMAIN-CM')



if __name__ == '__main__':
    main()
