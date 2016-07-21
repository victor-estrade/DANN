#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import time

import numpy as np

from logs import log_fname, new_logger, empty_logger
from datasets.utils import AD, AttributeDict, Dataset
# AD & Dataset are aliasies for AttributeDict

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
    Helper function interating over the given inputs

    Params
    ------
        inputs: the data (numpy array or tuple of numpy array)
        targets: the target values (numpy array or tuple of numpy array)
        batchsize: the batch size (int)
        shuffle (default=True): whether or not the data should be shuffled
    
    Return
    ------
        (input_slice, target_slice) as a generator
    """
    data = tuple()
    # Handle multiple or None inputs :
    if inputs is None:
        input_size = None
    elif isinstance(inputs, tuple) or isinstance(inputs, list):
        input_size = len(inputs[0])
        size = input_size
        for inpt in inputs:
            assert input_size == len(inpt), 'Each input should have the same number of examples'
        data += tuple(inputs)
    else:
        input_size = len(inputs)
        size = input_size
        data += (inputs,)

    # Handle multiple or None targets :
    if targets is None:
        target_size = None
    elif isinstance(targets, tuple) or isinstance(targets, list):
        target_size = len(targets[0])
        size = target_size
        for target in targets:
            assert target_size == len(target), 'Each target should have the same number of examples'
        data += tuple(targets)
    else:
        target_size = len(targets)
        size = target_size
        data += (targets,)

    if target_size is not None and input_size is not None:
        assert target_size == input_size, 'inputs and targets should have the same number of examples'

    if shuffle:
        indices = np.arange(size)
        np.random.shuffle(indices)
    for start_idx in range(0, size - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(arr[excerpt] for arr in data)



class NN(object):
    """
    Access the compiled functions :
    >>> nn[]
    """
    def __init__(self, end_layer, name='anonymous NN', logger=None):
        self.name = name
        self.end_layer = end_layer
        self.funs = AttributeDict()
        self.global_stats = AttributeDict()
        self.epoch_count = 0
        if logger is None:
            self.logger = empty_logger()
        else:
            self.logger = logger

    def compile(self, compiler, **kwargs):
        """
        Compile the output layer whose name is given.
        """
        funs = AttributeDict(compiler(self.end_layer, **kwargs))
        # funs is a dict containing compiled functions (train, valid, predict, etc)
        self.funs.update(funs)


    def _training_epoch(self, data):
        """
        Should do one epoch of training
        """
        epoch_stats = {'training '+stat_name:[] 
                       for stat_name in self.funs.train_description}
        # Training : (forward and backward propagation)
        # done with the iterative functions
        minibatches = iterate_minibatches(data['X_train'], data['y_train'], data['batchsize'], shuffle=True)
        # We train minibatch after minibatch
        for batch in minibatches:
            results = self.funs.train(*batch)
            # The first should be the loss
            # the second be accuracy
            for res, stat_name in zip(results, self.funs.train_description):
                epoch_stats['training '+stat_name].append(res)
        return epoch_stats

    def _validation_epoch(self, data):
        """
        Should do a validation epoch
        """
        epoch_stats = {'validation '+stat_name:[] 
                       for stat_name in self.funs.valid_description}
        # Training : (forward and backward propagation)
        # done with the iterative functions
        minibatches = iterate_minibatches(data['X_train'], data['y_train'], data['batchsize'], shuffle=True) 
        # We feed each part alternatively minibatch after minibatch
        for batch in minibatches:
            results = self.funs.valid(*batch)
            # The first should be the loss
            # the second be accuracy
            for res, stat_name in zip(results, self.funs.valid_description):
                epoch_stats['validation '+stat_name].append(res)
        return epoch_stats

    def train(self, data, num_epochs=20):
        """
        The train fuction (for the user)
        """
        # Init the stats of this training session.
        session_stats = {}
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({'training '+s:[] 
                 for s in self.funs.train_description})
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({'validation '+s:[] 
                 for s in self.funs.valid_description})

        for epoch in range(num_epochs):
            # We mesure the time it takes.
            # It helps to know how much time the code needs at runtime
            start_time = time.time()
            # Do the training (forward & backward on minibatches)
            train_stats = self._training_epoch(data)
            # Do the validation (forward & validation mesures on minibatches)
            valid_stats = self._validation_epoch(data)
            # Print time elapsed
            self.logger.info("Epoch {} of {} took {:.3f}s".format(
                self.epoch_count+epoch+1, self.epoch_count+num_epochs, time.time() - start_time))

            # Update the final_stats with the mean of the epoch stats
            for stat_name, stat_value in sorted(train_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)
            for stat_name, stat_value in sorted(valid_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)
        
        for stat_name, stat_value in session_stats.items():
            if stat_name in self.global_stats:
                self.global_stats[stat_name].extend(stat_value)
            else:
                self.global_stats[stat_name] = stat_value
        self.epoch_count += num_epochs
        return self

    def train_only(self, data, num_epochs=20):
        """
        The train fuction (for the user)
        """
        # Init the stats of this training session.
        session_stats = {}
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({'training '+s:[] 
                 for s in self.funs.train_description})

        for epoch in range(num_epochs):
            # We mesure the time it takes.
            # It helps to know how much time the code needs at runtime
            start_time = time.time()
            # Do the training (forward & backward on minibatches)
            train_stats = self._training_epoch(data)
            # Print time elapsed
            self.logger.info("Epoch {} of {} took {:.3f}s".format(
                self.epoch_count+epoch+1, self.epoch_count+num_epochs, time.time() - start_time))

            # Update the final_stats with the mean of the epoch stats
            for stat_name, stat_value in sorted(train_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)

        for stat_name, stat_value in session_stats.items():
            if stat_name in self.global_stats:
                self.global_stats[stat_name].extend(stat_value)
            else:
                self.global_stats[stat_name] = stat_value
        self.epoch_count += num_epochs
        return self


class CNN(object):
    """
    Access the compiled functions :
    >>> nn[]
    """
    def __init__(self, name='anonymous CNN', logger=None):
        self.name = name
        self.end_layers = AttributeDict()
        self.parts = AttributeDict()
        self.global_stats = AttributeDict()
        self.epoch_count = 0
        if logger is None:
            self.logger = empty_logger()
        else:
            self.logger = logger

    def __getitem__(self, name):
        if name in self.parts:
            return self.parts[name]
        else:
            raise AttributeError("No such compiled parts: " + name)

    def __setitem__(self, name, value):
        self.end_layers[name] = value

    def add_output(self, name, layer):
        self.end_layers[name] = layer

    def __delitem__(self, part_name):
        if part_name in self.parts:
            del self.parts[part_name]
        else:
            raise AttributeError("No such compiled parts: " + part_name)

    def compile(self, name, compiler, **kwargs):
        """
        Compile the output layer whose name is given.
        """
        if name in self.end_layers:
            funs = AttributeDict(compiler(self.end_layers[name], **kwargs))
            if name in self.parts:
                # funs is a dict containing compiled functions (train, valid, predict, etc)
                self.parts[name].update(funs)
            else:
                self.parts[name] = funs

        else:
            raise AttributeError("No such output layer: " + name)

    def _training_epoch(self, datas, names):
        """
        Should do one epoch of training
        """
        epoch_stats = {name+' training '+stat_name:[] 
                       for name in names 
                       for stat_name in self.parts[name].train_description}
        # Training : (forward and backward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_train'], data['y_train'], data['batchsize'], shuffle=True) 
                        for data in datas)
        # We train each part alternatively minibatch after minibatch
        for minibatches in zip(*batches): # draw the next minibatches
            for batch, name in zip(minibatches, names):
                results = self.parts[name].train(*batch)
                # The first should be the loss
                # the second be accuracy
                for res, stat_name in zip(results, self.parts[name].train_description):
                    epoch_stats[name+' training '+stat_name].append(res)
        return epoch_stats

    def _validation_epoch(self, datas, names):
        """
        Should do a validation epoch
        """
        epoch_stats = {name+' validation '+stat_name:[] 
                       for name in names 
                       for stat_name in self.parts[name].valid_description}
        # Training : (forward and backward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_train'], data['y_train'], data['batchsize'], shuffle=True) 
                        for data in datas)
        # We feed each part alternatively minibatch after minibatch
        for minibatches in zip(*batches): # draw the next minibatches
            for batch, name in zip(minibatches, names):
                results = self.parts[name].valid(*batch)
                # The first should be the loss
                # the second be accuracy
                for res, stat_name in zip(results, self.parts[name].valid_description):
                    epoch_stats[name+' validation '+stat_name].append(res)
        return epoch_stats

    def train(self, datas, names, num_epochs=20):
        """
        The train fuction (for the user)
        """
        # Init the stats of this training session.
        session_stats = {}
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({name+' training '+s:[] 
                 for name in names 
                 for s in self.parts[name].train_description})
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({name+' validation '+s:[] 
                 for name in names 
                 for s in self.parts[name].valid_description})

        for epoch in range(num_epochs):
            # We mesure the time it takes.
            # It helps to know how much time the code needs at runtime
            start_time = time.time()
            # Do the training (forward & backward on minibatches)
            train_stats = self._training_epoch(datas, names)
            # Do the validation (forward & validation mesures on minibatches)
            valid_stats = self._validation_epoch(datas, names)
            # Print time elapsed
            self.logger.info("Epoch {} of {} took {:.3f}s".format(
                self.epoch_count+epoch+1, self.epoch_count+num_epochs, time.time() - start_time))

            # Update the final_stats with the mean of the epoch stats
            for stat_name, stat_value in sorted(train_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)
            for stat_name, stat_value in sorted(valid_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)
        
        for stat_name, stat_value in session_stats.items():
            if stat_name in self.global_stats:
                self.global_stats[stat_name].extend(stat_value)
            else:
                self.global_stats[stat_name] = stat_value
        self.epoch_count += num_epochs
        return self


    def train_only(self, datas, names, num_epochs=20):
        """
        The train fuction (for the user)
        """
        # Init the stats of this training session.
        session_stats = {}
        # Prepare the empty lists that will recieve the stats' mean for each epoch
        session_stats.update({name+' training '+s:[] 
                 for name in names 
                 for s in self.parts[name].train_description})

        for epoch in range(num_epochs):
            # We mesure the time it takes.
            # It helps to know how much time the code needs at runtime
            start_time = time.time()
            # Do the training (forward & backward on minibatches)
            train_stats = self._training_epoch(datas, names)
            # Print time elapsed
            self.logger.info("Epoch {} of {} took {:.3f}s".format(
                self.epoch_count+epoch+1, self.epoch_count+num_epochs, time.time() - start_time))

            # Update the final_stats with the mean of the epoch stats
            for stat_name, stat_value in sorted(train_stats.items()):
                if stat_value:
                    mean_value = np.mean(stat_value)
                    self.logger.info('   {:30} : {:.6f}'.format(stat_name, mean_value))
                    session_stats[stat_name].append(mean_value)

        for stat_name, stat_value in session_stats.items():
            if stat_name in self.global_stats:
                self.global_stats[stat_name].extend(stat_value)
            else:
                self.global_stats[stat_name] = stat_value
        self.epoch_count += num_epochs
        return self
