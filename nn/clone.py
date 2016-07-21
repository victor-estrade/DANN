# -*- coding: utf-8 -*-
from __future__ import division, print_function

import lasagne


def _clone_DenseLayer(layer, input_layer=None):
    if not isinstance(layer, lasagne.layers.DenseLayer):
        raise ValueError("The given layer should be a lasagne.layers.DenseLayer,"
                         "{} given".format(layer.__class__))
    else:
        if input_layer is None:
            input_layer = layer.input_layer
        return lasagne.layers.DenseLayer(input_layer,
                                        num_units=layer.num_units,
                                        nonlinearity=layer.nonlinearity,
                                        W=layer.W, b=layer.b)


# Clone factory
clonable_layers = {
    lasagne.layers.DenseLayer: _clone_DenseLayer
}


def clone_layer(layer, input_layer=None):
    if any([isinstance(layer, key) for key in clonable_layers.keys()]):
        return clonable_layers[layer.__class__](layer, input_layer)
    else:
        raise NotImplementedError('{} is not a clonable layer (yet)'.format(layer.__class__))
