#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from dann import main
from logs import new_logger, log_fname
from utils import pop_last_line

__author__ = 'Estrade Victor'


if __name__ == '__main__':
    np.savetxt('TODO.txt', np.logspace(-3, 0, num=5))
    
    todo = pop_last_line('TODO.txt')
    while todo:
        hp_lambda = float(todo)
        model = 'small'
        title = 'S-T-{}-lambda-{:.3f}-'.format(model, hp_lambda)
        for i in range(3):
            f_log = log_fname(title)
            logger = new_logger(f_log)
            
            stats = main(model=model, num_epochs=40, hp_lambda=hp_lambda, 
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

