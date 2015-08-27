import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *
import itertools


class FFN_vat(ModelSLBase):
    """
    Feedforward neural network with virtual adversarial training
    """

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)
        self.shared_vars.save(self.filename + '_vars')

    def __init__(self, data, hp):
        super(FFN_vat, self).__init__(self.__class__.__name__, data, hp)
        
        self.epsilon = 0.001

        self.params = Parameters()
        self.shared_vars = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        
        dropout_rate = 0.3
        n_h1 = 1200
        n_h2 = 1000
        n_h3 = 800

        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
            self.shared_vars.load(self.filename + '_vars')
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h1), scale=scale)
                b_h = shared_normal((n_h1,), scale=0)
                w_h2 = shared_normal((n_h1, n_h2), scale=scale)
                b_h2 = shared_normal((n_h2,), scale=0)
                w_h3 = shared_normal((n_h2, n_h3), scale=scale)
                b_h3 = shared_normal((n_h3,), scale=0)
                w_o = shared_normal((n_h3, n_y), scale=scale)

        def model(X, params, shared_vars, p_drop_hidden):

            h = dropout(rectify(T.dot(X, params.w_h) + params.b_h ), p_drop_hidden)
            h2 = dropout(rectify(T.dot(h, params.w_h2) + params.b_h2), p_drop_hidden)
            h3 = dropout(rectify(T.dot(h2, params.w_h3) + params.b_h3), p_drop_hidden)

            py_x = softmax(T.dot(h3, params.w_o))
            return py_x
        
        # Train
        add_updates = []

        adv_est_noise = 1e-6
        sl_noise = 0.2
        adv_noise = 2.0
        adv_cost_coeff = 1.0

        #clean_py_x = model(self.X + gaussian(self.X.shape, sl_noise), self.params, self.shared_vars, dropout_rate)
        clean_py_x = model(self.X, self.params, self.shared_vars, dropout_rate)

        adv_X = normalize(gaussian(self.X.shape, 1.0))

        adv_py_x = model(self.X + adv_X*adv_est_noise, self.params, self.shared_vars, dropout_rate)
        cost_adv = T.sum(T.nnet.categorical_crossentropy(adv_py_x, clean_py_x))
        
        adv_X = T.grad(cost=cost_adv, wrt=self.X)
        adv_X = theano.gradient.disconnected_grad(adv_X)
        adv_X = normalize(adv_X)
        
        adv_py_x = model(self.X + adv_X*adv_noise, self.params, self.shared_vars, dropout_rate)
        noise_py_x = model(self.X + gaussian(self.X.shape, sl_noise), self.params, self.shared_vars, dropout_rate)
        noise_py_x_hat = theano.gradient.disconnected_grad(noise_py_x)

        sl_cost = T.sum(T.nnet.categorical_crossentropy(noise_py_x, self.Y))
        adv_cost = T.sum(T.nnet.categorical_crossentropy(adv_py_x, noise_py_x_hat))

        cost = sl_cost + adv_cost*adv_cost_coeff

        # Test
        pyx = model(self.X, self.params, self.shared_vars, 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))
        
        self.compile(cost, error_map_pyx, add_updates, [adv_X])

        