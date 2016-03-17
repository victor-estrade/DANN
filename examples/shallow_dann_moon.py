
# coding: utf-8

# In[5]:

from __future__ import division

import theano
import theano.tensor as T
import lasagne

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_moon
from rgl import ReverseGradientLayer
from logs import log_fname, new_logger


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


def compiler_sgd_mom(lr=1, mom=.9) : 
    
    def get_fun(output_layer, lr=1, mom=.9, target_var=T.ivector('target')):

        input_var = lasagne.layers.get_all_layers(output_layer)[0].input_var
        pred = lasagne.layers.get_output(output_layer)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
        params = lasagne.layers.get_all_params(output_layer)
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
        updates = lasagne.updates.apply_momentum(updates, params, momentum=mom)
        acc = T.mean(T.eq(T.argmax(pred, axis=1), target_var))
        train_function = theano.function([input_var, target_var], [loss, acc], 
            updates=updates, allow_input_downcast=True)
        
        pred = lasagne.layers.get_output(output_layer, deterministic=True)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(pred, target_var))
        label = T.argmax(pred, axis=1)
        acc = T.mean(T.eq(label, target_var))
        predict_function = theano.function([input_var], [label], allow_input_downcast=True)
        valid_function = theano.function([input_var, target_var], [loss, acc], allow_input_downcast=True)
        proba_function = theano.function([input_var], [pred], allow_input_downcast=True)

        return train_function, predict_function, valid_function, proba_function
    
    return(lambda output_layer: get_fun(output_layer, lr=lr, mom=mom))


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



class ShallowDANN(object) :
    
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
            source_batches = iterate_minibatches(source['X_train'], source['y_train'], source['batchsize'])
            domain_batches = iterate_minibatches(domain['X_train'], domain['y_train'], domain['batchsize'])
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


def main(hp_lambda=0.0, num_epochs=50, angle=-35, label_rate=1, domain_rate=1):
    """
    The main function.
    """
    # Moon Dataset
    data_name = 'moon'
    batchsize = 32
    source_data, target_data, domain_data = load_moon(angle=angle)

    # Set up the training :
    datas = [source_data, domain_data, target_data]

    model = '1DR'

    title = '{}-lambda-{:.4f}-{}'.format(model, hp_lambda, data_name)
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
    dann = ShallowDANN(3, 2, input_layer, hp_lambda=hp_lambda)

    logger.info('Compiling functions')
    dann.compile_label(compiler_sgd_mom(lr=label_rate, mom=0))
    dann.compile_domain(compiler_sgd_mom(lr=domain_rate, mom=0))
    
    # Train the NN
    dann.training(source_data, domain_data, num_epochs=num_epochs)
    
    # Plot boundary :
    X = np.vstack([source_data['X_train'], source_data['X_val'], source_data['X_test'], ])
    y = np.hstack([source_data['y_train'], source_data['y_val'], source_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, dann.proba_label)
    plt.title('Moon bounds')
    plt.savefig('fig/moon-bound.png')
    plt.clf() # Clear plot window
    
    X = np.vstack([target_data['X_train'], target_data['X_val'], target_data['X_test'], ])
    y = np.hstack([target_data['y_train'], target_data['y_val'], target_data['y_test'], ])
    colors = 'rb'
    plot_bound(X, y, dann.proba_label)
    plt.title('Moon rotated bounds')
    plt.savefig('fig/moon-rot-bound.png')
    plt.clf() # Clear plot window


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
    parser.add_argument(
        '--domain-rate', help="The learning rate of the domain part of the neural network ",
        default=1, type=float, dest='domain_rate')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    num_epochs = args.num_epochs
    hp_lambda = args.hp_lambda
    angle = args.angle
    label_rate = args.label_rate
    domain_rate = args.domain_rate
    main(hp_lambda=hp_lambda,  num_epochs=num_epochs, angle=angle,
        label_rate=label_rate, domain_rate=domain_rate,)
