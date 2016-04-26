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

from datasets.mnist import load_mnist
from datasets.utils import mirror_dataset, make_domain_dataset
from logs import log_fname, new_logger
from nn.rgl import ReverseGradientLayer
from nn.block import Dense, Classifier, adversarial
from nn.compilers import squared_error_sgd_mom, crossentropy_sgd_mom
from nn.training import Trainner, training
from utils import plot_bound, iterate_minibatches

# raise NotImplementedError('Coding in progress')


# http://stackoverflow.com/questions/25886374/pdist-for-theano-tensor
# Tested and approved
X = T.fmatrix('X')
Y = T.fmatrix('Y')
translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
euclidiean_distances = (translation_vectors ** 2).sum(2)
f_euclidean = theano.function([X, Y], euclidiean_distances, allow_input_downcast=True)


def kclosest(X, Y, k, batchsize=None):
    """
    Computes for each sample from X the k-closest samples in Y and return 
    their index.

    Params
    ------
        X: (numpy array [n_sample, n_feature])
        Y: (numpy array [n_sample, n_feature])
        k: (int)
    Return
    ------
        kclosest : (numpy array [n_sample, k]) the ordered index of 
            the k-closest instances from Y to X samples
    """
    assert X.shape == Y.shape
    N = X.shape[0]
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)

    if batchsize is None:
        dist = f_euclidean(X, Y)
    else:
        dist = np.empty((N, N), dtype=theano.config.floatX)
        batch = np.arange(0, N+batchsize, batchsize)
        for excerpt_X in (slice(i0, i1) for i0, i1 in zip(batch[:-1], batch[1:])):
            dist[excerpt_X] = f_euclidean(X[excerpt_X], Y)
    kbest = np.argsort(dist, axis=1)[:, :k]
    return kbest


def realign(X_out, X_trg, y, k=5, batchsize=None):
    counter = np.zeros(X_out.shape[0], dtype=int)
    idx = np.empty_like(y, dtype=int)
    for label in np.unique(y):
        # Get the examples of the right label
        idx_label = np.where(y==label)[0]

        # Get the k-closest index ... shape = ... ça va pas du tout !
        idx_label2 = kclosest(X_out[idx_label], X_trg[idx_label], k, batchsize=batchsize)
        
        for i1, i2 in zip(idx_label, idx_label2):
            # i2 is an index array of shape (k,) with the sorted closest example index 
            # (of the sorted single class array)
            # Then idx_label[i2] are the sorted original index of the k-closest examples
            i = idx_label[i2[np.argmin(counter[idx_label[i2]])]]
            # i contains the chosen one, in the (k-)clostest example, with the minimum counter
            counter[i] = counter[i]+1
            idx[i1] = i
    return idx


def batchpad(batchsize, output_shape, dtype=None):
    """Re-batching decorator
    """
    def decoreted(func):
        def wrapper(X, *args, **kwargs):
            if dtype is None:
                dtype2 = X.dtype
            else:
                dtype2 = dtype
            
            N = X.shape[0]
            
            if output_shape is None:
                shape = X.shape
            else:
                shape = tuple( out_s if out_s is not None else X_s for out_s, X_s in zip(output_shape, X.shape))

            result = np.empty(shape, dtype=dtype2)
            batch = np.arange(0, N+batchsize, batchsize)
            for excerpt_X in (slice(i0, i1) for i0, i1 in zip(batch[:-2], batch[1:])):
                result[excerpt_X] = func(X[excerpt_X], *args, **kwargs)
            
            last_excerpt = slice(batch[-2], batch[-1])
            X = X[last_excerpt]
            n_sample = X.shape[0]
            X = np.vstack([X, np.zeros((batchsize-X.shape[0],)+X.shape[1:])])
            X = func(X, *args, **kwargs)
            result[last_excerpt] = X[:n_sample]
            
            return result
        return wrapper
    return decoreted


def preprocess(data, trainer, epoch):
    X = data['X_train']

    @batchpad(data['batchsize'], X.shape, X.dtype)
    def f_output(X, trainer):
        return trainer.output(X)[0]
    
    X_out = f_output(X, trainer)
    X_trg = data['y_train']
    data['X_train'] = X[realign(X_out, X_trg, data['labels'], k=200, batchsize=100)]


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
        default=2, type=int, dest='num_epochs')
    parser.add_argument(
        '--batchsize', help='The mini-batch size',
        default=500, type=int, dest='batchsize')
    parser.add_argument(
        '--lambda', help='Value of the lambda_D param of the Reversal Gradient Layer',
        default=0., type=float, dest='hp_lambda')
    parser.add_argument(
        '--label-rate', help="The learning rate of the label part of the neural network ",
        default=2, type=float, dest='label_rate')
    parser.add_argument(
        '--label-mom', help="The learning rate momentum of the label part of the neural network ",
        default=0.9, type=float, dest='label_mom')
    parser.add_argument(
        '--domain-rate', help="The learning rate of the domain part of the neural network ",
        default=1, type=float, dest='domain_rate')
    parser.add_argument(
        '--domain-mom', help="The learning rate momentum of the domain part of the neural network ",
        default=0.9, type=float, dest='domain_mom')

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
    data_name = 'MNISTMirror'
    model = 'AlignLearnCorrector'
    title = '{}-{}-lambda-{:.2e}'.format(data_name, model, hp_lambda)

    #=========================================================================
    # Load, Generate the datasets
    #=========================================================================
    # Load MNIST Dataset
    source_data = load_mnist(batchsize=batchsize)
    target_data = mirror_dataset(source_data)
    domain_data = make_domain_dataset([source_data, target_data])

    corrector_data = dict(target_data)
    corrector_data.update({
        'y_train': source_data['X_train'],
        'y_val': source_data['X_val'],
        'y_test': source_data['X_test'],
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
    input_var = T.tensor3('inputs')
    target_var = T.tensor3('targets')
    shape = (batchsize, 28, 28)

    # Build the layers
    input_layer = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    src_layer = lasagne.layers.InputLayer(shape=shape, input_var=T.tensor3('src'))
    feature = lasagne.layers.DenseLayer(
                    input_layer,
                    num_units=np.prod(shape[1:]),
                    nonlinearity=None,
                    # W=lasagne.init.Uniform(range=0.01, std=None, mean=0.0),
                    )
    reshaper = lasagne.layers.ReshapeLayer(feature, (-1,) + shape[1:])
    output_layer = reshaper

    # Compilation
    logger.info('Compiling functions')
    corrector_trainner = Trainner(squared_error_sgd_mom(output_layer, lr=label_rate, mom=label_mom, target_var=target_var), 
                                  'corrector',)
    corrector_trainner.preprocess = preprocess

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

    # Plot some sample images:
    fig = plt.figure()
    n_sample = 4
    rand = np.random.RandomState()
    for n in range(n_sample):
        i = rand.randint(source_data['X_test'].shape[0])
        sample_src = source_data['X_test'][i]
        sample_trg = target_data['X_test'][i]
        sample_corrected = corrector_trainner.output(target_data['X_test'][i][np.newaxis])
        sample_corrected = np.array(sample_corrected).reshape((28,28))
        ax = fig.add_subplot(n_sample, 3, n*3+1)
        ax.axis('off')
        ax.imshow(sample_src, cmap='Greys_r')
        ax.set_title('Source image')
        ax = fig.add_subplot(n_sample, 3, n*3+2)
        ax.axis('off')
        ax.imshow(sample_trg, cmap='Greys_r')
        ax.set_title('Target image')
        ax = fig.add_subplot(n_sample, 3, n*3+3)
        ax.axis('off')
        ax.imshow(sample_corrected, cmap='Greys_r')
        ax.set_title('Corrected image')
    fig.savefig('fig/{}-sample.png'.format(title))
    plt.close(fig) # Clear plot window

    # Plot the weights of the corrector
    W = feature.W.get_value()
    plt.imshow(W, interpolation='nearest', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('fig/{}-Weights.png'.format(title))
    

if __name__ == '__main__':
    main()
