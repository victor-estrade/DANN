# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np


def batchpad(batchsize, output_shape, dtype=None):
    """
    Re-batching decorator.

    """
    def decoreted(func):
        def wrapper(X, *args, **kwargs):
        	# dtype is not in the local namespace. Have to init a dtype2 from it. (Why?)
            if dtype is None:
                dtype2 = X.dtype
            else:
                dtype2 = dtype
            
            N = X.shape[0]
            
            if output_shape is None:
                shape = X.shape
            else:
                shape = tuple( out_s if out_s is not None else X_s 
                              for out_s, X_s in zip(output_shape, X.shape))

            result = np.empty(shape, dtype=dtype2)
            batch = np.arange(0, N, batchsize)
            for excerpt_X in (slice(i0, i1) for i0, i1 in zip(batch[:-1], batch[1:])):
                result[excerpt_X] = func(X[excerpt_X], *args, **kwargs)
            
            last_excerpt = slice(batch[-1], N)
            X = X[last_excerpt]
            n_sample = X.shape[0]
            X = np.vstack([X, np.zeros((batchsize-X.shape[0],)+X.shape[1:])])
            X = func(X, *args, **kwargs)
            result[last_excerpt] = X[:n_sample]
            
            return result
        return wrapper
    return decoreted