#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from logs import extract, multi_extract


if __name__ == '__main__':
    log_file = 'log/test.log'
    sep = r'\s*:\s'
    # Extract from the log file
    
    train_loss = r'training\sloss'
    source_validation_loss = r'source\svalidation\sloss'
    source_validation_accuracy = r'source\svalidation\saccuracy'
    target_validation_loss = r'target\svalidation\sloss'
    target_validation_accuracy = r'target\svalidation\saccuracy'
    source_test_loss = r'source\stest\sloss'
    source_test_accuracy = r'source\stest\saccuracy'
    target_test_loss = r'target\stest\sloss'
    target_test_accuracy = r'target\stest\saccuracy'

    names = [train_loss,
            source_validation_loss,
            source_validation_accuracy,
            target_validation_loss,
            target_validation_accuracy,
            source_test_loss,
            source_test_accuracy,
            target_test_loss,
            target_test_accuracy,
            ]
    seps = [sep for name in names]
    values = multi_extract(log_file, names, seps)
    plt.plot(values[source_validation_accuracy], label='source', c='blue')
    plt.plot(values[target_validation_accuracy], label='target', c='red')
    plt.axhline(y=values[source_test_accuracy][0], c='blue')
    plt.axhline(y=values[target_test_accuracy][0], c='red')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.show()

    