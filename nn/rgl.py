#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import theano
import lasagne

import theano.tensor as T
import numpy as np
"""
used :
http://stackoverflow.com/questions/33879736/can-i-selectively-invert-theano-gradients-during-backpropagation
for the implementation of the operation and the layer.

used :
https://groups.google.com/forum/#!topic/lasagne-users/8gFgB8c-EYk
to get updatable hp_lambda.
example:
>>> rglayer = ReverseGradientLayer(a_layer, 1.0)
>>>for ... in ...:
>>>   if ...:
>>>      rglayer.hp_lambda.set_value(rglayer.hp_lambda.get_value() * decay)
>>>   ...
"""


class ReverseGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


class ReverseGradientLayer(lasagne.layers.Layer):
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.hp_lambda = theano.shared(np.array(hp_lambda, dtype=theano.config.floatX))
        self.op = ReverseGradient(hp_lambda)

    def get_output_for(self, input, **kwargs):
        return self.op(input)

    def set_lambda(self, hp_lambda):
        """
        Untested yet
        """
        self.hp_lambda.set_value(np.array(hp_lambda, dtype=theano.config.floatX))

    def decay_lambda(self, decay):
        """
        Untested yet
        """
        self.hp_lambda.set_value(self.hp_lambda.get_value() * np.array(decay, dtype=theano.config.floatX))
