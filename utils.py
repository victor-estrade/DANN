# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt


def reverse_lines(fname):
    """
    Reverse the lines order in a file

    Params
    ------
        fname: the file name to be reversed
    """
    with open(fname, 'r') as file:
        lines = file.readlines()
    with open(fname, 'w') as file:
        file.write(''.join(reversed(lines)))


def pop_last_line(fname):
    """
    Get and remove the last line from the given file

    Params
    ------
        fname: the file name from which to extract the last line
    Return
    ------
        line: le last line of the given file
    """
    pos = []
    with open(fname, 'rw+') as file:
        pos.append(file.tell())
        line = file.readline()
        while line:
            pos.append(file.tell())
            line = file.readline()
        if len(pos) > 1:
            file.seek(pos[-2])
            line = file.readline()
            file.truncate(pos[-2])
    return line


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Helper function interating over the given inputs

    Params
    ------
        inputs: the data (numpy array or tuple of numpy array)
        targets: the target values (numpy array)
        batchsize: the batch size (int)
        shuffle (default=False): whether or not the data should be shuffled
    
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
