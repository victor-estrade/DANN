#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import time

import numpy as np

from logs import log_fname, new_logger, empty_logger
from utils import iterate_minibatches


class Trainner(object):
    def __init__(self, funs, name='trainer'):
        super(Trainner, self).__init__()
        self.name = name
        # Add the compiled functions to the object
        # by adding dynamic property to this object
        self.__dict__.update(funs)
    
    def preprocess(self, *args, **kwargs):
        pass


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
        logger = empty_logger()

    logger.info("Starting training...")
    final_stats = {}
    final_stats.update({trainer.name+' training loss': [] for trainer in trainers})
    final_stats.update({trainer.name+' valid loss': [] for trainer in trainers})
    final_stats.update({tester.name+' valid loss': [] for tester in testers})

    final_stats.update({(trainer.name+str(i)+' training acc' if trainer.train.n_returned_outputs > 2
                            else trainer.name+' training acc'): []
        for trainer in trainers for i in range(trainer.train.n_returned_outputs-1)})
    final_stats.update({(trainer.name+str(i)+' valid acc' if trainer.train.n_returned_outputs > 2
                            else trainer.name+' valid acc'): []
        for trainer in trainers for i in range(trainer.train.n_returned_outputs-1)})
    final_stats.update({(tester.name+str(i)+' valid acc' if tester.train.n_returned_outputs > 2
                            else tester.name+' valid acc'): []
        for tester in testers for i in range(tester.train.n_returned_outputs-1)})
    # final_stats.update({trainer.name+' valid acc': [] for trainer in trainers})
    # final_stats.update({tester.name+' valid acc': [] for tester in testers})

    for epoch in range(num_epochs):
        # Prepare the statistics
        start_time = time.time()
        stats = { key:[] for key in final_stats.keys()}

        # Do some trainning preparations :
        for data, trainer in zip(train_data+test_data, trainers+testers):
            trainer.preprocess(data, trainer, epoch)

        # Training : (forward and backward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_train'], data['y_train'], data['batchsize'], shuffle=True) 
                        for data in train_data)
        for minibatches in zip(*batches):
            for batch, trainer in zip(minibatches, trainers):
                # X, y = batch
                res = trainer.train(*batch)
                loss, acc = res.pop(0), res
                # The first should be the loss
                stats[trainer.name+' training loss'].append(loss)
                # If we are in a normal case, res is only one accuracy
                if len(acc) == 1:
                    stats[trainer.name+' training acc'].append(acc*100)
                else: # Else we have multiple accuracies
                    for i, a in enumerate(acc):
                        stats[trainer.name+str(i)+' training acc'].append(a*100)

        # Validation (forward propagation)
        # done with the iterative functions
        batches = tuple(iterate_minibatches(data['X_val'], data['y_val'], data['batchsize']) 
                        for data in train_data+test_data)
        for minibatches in zip(*batches):
            for batch, valider in zip(minibatches, trainers+testers):
                # X, y = batch
                res = valider.valid(*batch)
                loss, acc = res.pop(0), res
                # The first should be the loss
                stats[valider.name+' valid loss'].append(loss)
                # If we are in a normal case, res is only one accuracy
                if len(acc) == 1:
                    stats[valider.name+' valid acc'].append(acc*100)
                else: # Else we have multiple accuracies
                    for i, a in enumerate(acc):
                        stats[valider.name+str(i)+' valid acc'].append(a*100)
        
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        for stat_name, stat_value in sorted(stats.items()):
            if stat_value:
                mean_value = np.mean(stat_value)
                logger.info('   {:30} : {:.6f}'.format(
                    stat_name, mean_value))
                final_stats[stat_name].append(mean_value)

    return final_stats
