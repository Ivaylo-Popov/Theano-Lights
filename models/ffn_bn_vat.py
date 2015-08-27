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


class FFN_bn_vat(ModelSLBase):
    """
    Virtual adversarial training with batch normalization
    """

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)
        self.shared_vars.save(self.filename + '_vars')

    def __init__(self, data, hp):
        super(FFN_bn_vat, self).__init__(self.__class__.__name__, data, hp)
        
        self.epsilon = 0.001

        self.params = Parameters()
        self.shared_vars = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h1 = 1200
        n_h2 = 800
        n_h3 = 600
        n_h4 = 400
        scale = hp.init_scale

        dropout_rate = 0.3

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
            self.shared_vars.load(self.filename + '_vars')
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h1), scale=scale)
                gamma = shared_uniform((n_h1,), range=[0.95, 1.05])
                b_h = shared_zeros((n_h1,))

                w_h2 = shared_normal((n_h1, n_h2), scale=scale)
                gamma2 = shared_uniform((n_h2,), range=[0.95, 1.05])
                b_h2 = shared_zeros((n_h2,))

                w_h3 = shared_normal((n_h2, n_h3), scale=scale)
                gamma3 = shared_uniform((n_h3,), range=[0.95, 1.05])
                b_h3 = shared_zeros((n_h3,))

                w_h4 = shared_normal((n_h3, n_h4), scale=scale)
                gamma4 = shared_uniform((n_h4,), range=[0.95, 1.05])
                b_h4 = shared_zeros((n_h4,))

                w_o = shared_normal((n_h4, n_y), scale=scale)

            with self.shared_vars:
                m_shared = shared_zeros((1, n_h1), broadcastable=(True, False))
                v_shared = shared_zeros((1, n_h1), broadcastable=(True, False))
                m_shared2 = shared_zeros((1, n_h2), broadcastable=(True, False))
                v_shared2 = shared_zeros((1, n_h2), broadcastable=(True, False))
                m_shared3 = shared_zeros((1, n_h3), broadcastable=(True, False))
                v_shared3 = shared_zeros((1, n_h3), broadcastable=(True, False))
                m_shared4 = shared_zeros((1, n_h4), broadcastable=(True, False))
                v_shared4 = shared_zeros((1, n_h4), broadcastable=(True, False))
        
        def batch_norm(X, m_shared, v_shared, test, add_updates):
            if X.ndim > 2:
                output_shape = X.shape
                X = X.flatten(2)
 
            if test is False:
                m = T.mean(X, axis=0, keepdims=True)
                v = T.sqrt(T.var(X, axis=0, keepdims=True) + self.epsilon)
                
                if not add_updates is None:
                    mulfac = 1.0/100
                    add_updates.append((m_shared, (1.0-mulfac)*m_shared + mulfac*m))
                    add_updates.append((v_shared, (1.0-mulfac)*v_shared + mulfac*v))
            else:
                m = m_shared
                v = v_shared

            X_hat = (X - m) / v
 
            if X.ndim > 2:
                X_hat = T.reshape(X_hat, output_shape)
            return X_hat

        def model(X, params, sv, p_drop_hidden, test, add_updates):
            h = batch_norm(T.dot(X, params.w_h), sv.m_shared, sv.v_shared, test, add_updates)
            h = params.gamma * h + params.b_h
            h = dropout(rectify(h), p_drop_hidden)

            h2 = batch_norm(T.dot(h, params.w_h2), sv.m_shared2, sv.v_shared2, test, add_updates)
            h2 = params.gamma2 * h2 + params.b_h2
            h2 = dropout(rectify(h2), p_drop_hidden)

            h3 = batch_norm(T.dot(h2, params.w_h3), sv.m_shared3, sv.v_shared3, test, add_updates)
            h3 = params.gamma3 * h3 + params.b_h3
            h3 = dropout(rectify(h3), p_drop_hidden)

            h4 = batch_norm(T.dot(h3, params.w_h4), sv.m_shared4, sv.v_shared4, test, add_updates)
            h4 = params.gamma4 * h4 + params.b_h4
            h4 = dropout(rectify(h4), p_drop_hidden)

            py_x = softmax(T.dot(h4, params.w_o))
            return py_x
        
        add_updates = []

        x = self.X + gaussian(self.X.shape, 0.1)
        py_x = model(x, self.params, self.shared_vars, dropout_rate, False, add_updates)
        cost = -T.sum(self.Y * T.log(py_x))
        
        # VAT
        adv_cost_coeff = 1.0
        adv_est_noise = 1e-6
        adv_noise = 3.0
        adv_power_iter = 1

        if adv_cost_coeff > 0:
            adv_X = normalize(gaussian(self.X.shape, 1.0))
        
            for power_it in xrange(0, adv_power_iter):
                d = adv_X*adv_est_noise
                adv_est_py_x = model(x + d, self.params, self.shared_vars, dropout_rate, False, None)
                cost_adv = -T.sum(py_x * T.log(adv_est_py_x))
                adv_X = T.grad(cost=cost_adv, wrt=d)
                adv_X = normalize(theano.gradient.disconnected_grad(adv_X))
        
            adv_py_x = model(x + adv_X*adv_noise, self.params, self.shared_vars, dropout_rate, False, None)
            py_x_hat = theano.gradient.disconnected_grad(py_x)
            adv_cost = -T.sum(py_x_hat * T.log(adv_py_x))

            cost += adv_cost_coeff*adv_cost

        # -------------------------------------------------
        pyx = model(self.X, self.params, self.shared_vars, 0., True, None)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx, add_updates)

