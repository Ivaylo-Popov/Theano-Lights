import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import scipy.io
import time
import sys
import logging
import copy

from toolbox import *
from modelbase import *


class CNN(ModelSLBase):
    def __init__(self, data, hp):
        super(CNN, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 400

        self.params = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h = self.n_h
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w = shared_normal((32, 1, 3, 3), scale=scale)
                w2 = shared_normal((64, 32, 3, 3), scale=scale)
                w3 = shared_normal((128, 64, 3, 3), scale=scale)
                w4 = shared_normal((128 * 3 * 3, 625), scale=scale)
                w_o = shared_normal((625, n_y), scale=scale)
        
        def model(X, params, p_drop_conv, p_drop_hidden):
            l1a = rectify(conv2d(X, params.w, border_mode='full'))
            l1 = max_pool_2d(l1a, (2, 2))
            l1 = dropout(l1, p_drop_conv)

            l2a = rectify(conv2d(l1, params.w2))
            l2 = max_pool_2d(l2a, (2, 2))
            l2 = dropout(l2, p_drop_conv)

            l3a = rectify(conv2d(l2, params.w3))
            l3b = max_pool_2d(l3a, (2, 2))
            l3 = T.flatten(l3b, outdim=2)
            l3 = dropout(l3, p_drop_conv)

            l4 = rectify(T.dot(l3, params.w4))
            l4 = dropout(l4, p_drop_hidden)

            py_x = softmax(T.dot(l4, params.w_o))
            return py_x
        
        x = T.reshape(self.X, (-1, 1, 28, 28))

        noise_py_x = model(x, self.params, 0.2, 0.5)
        cost = T.sum(T.nnet.categorical_crossentropy(noise_py_x, self.Y))

        pyx = model(x, self.params, 0., 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx)