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


class FFN_bn(ModelSLBase):

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)
        self.shared_vars.save(self.filename + '_vars')

    def __init__(self, data, hp):
        super(FFN_bn, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 1600
        self.epsilon = 0.001

        self.params = Parameters()
        self.shared_vars = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h = self.n_h
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
            self.shared_vars.load(self.filename + '_vars')
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h), scale=scale)
                b_h = shared_normal((n_h,), scale=0)
                w_h2 = shared_normal((n_h, n_h), scale=scale)
                b_h2 = shared_normal((n_h,), scale=0)
                w_h3 = shared_normal((n_h, n_h), scale=scale)
                b_h3 = shared_normal((n_h,), scale=0)
                w_o = shared_normal((n_h, n_y), scale=scale)

                gamma = shared_uniform((self.n_h,), range=[0.95, 1.05])
                beta = shared_zeros((self.n_h,))
                gamma2 = shared_uniform((self.n_h,), range=[0.95, 1.05])
                beta2 = shared_zeros((self.n_h,))
                gamma3 = shared_uniform((self.n_h,), range=[0.95, 1.05])
                beta3 = shared_zeros((self.n_h,))

            with self.shared_vars:
                m_shared = shared_zeros((1, self.n_h), broadcastable=(True, False))
                v_shared = shared_zeros((1, self.n_h), broadcastable=(True, False))
                m_shared2 = shared_zeros((1, self.n_h), broadcastable=(True, False))
                v_shared2 = shared_zeros((1, self.n_h), broadcastable=(True, False))
                m_shared3 = shared_zeros((1, self.n_h), broadcastable=(True, False))
                v_shared3 = shared_zeros((1, self.n_h), broadcastable=(True, False))
        
        def batch_norm(X, gamma, beta, m_shared, v_shared, test, add_updates):
            if X.ndim > 2:
                output_shape = X.shape
                X = X.flatten(2)
 
            if test is False:
                m = T.mean(X, axis=0, keepdims=True)
                v = T.sqrt(T.var(X, axis=0, keepdims=True) + self.epsilon)
                
                mulfac = 1.0/100
                add_updates.append((m_shared, (1.0-mulfac)*m_shared + mulfac*m))
                add_updates.append((v_shared, (1.0-mulfac)*v_shared + mulfac*v))
            else:
                m = m_shared
                v = v_shared
            
            X_hat = (X - m) / v
            y = gamma*X_hat + beta
 
            if X.ndim > 2:
                y = T.reshape(y, output_shape)
            return y

        def model(X, params, shared_vars, p_drop_input, p_drop_hidden, test, add_updates):
            X = dropout(X, p_drop_input)
            
            h_bn = batch_norm(T.dot(X, params.w_h) + params.b_h, params.gamma, params.beta, shared_vars.m_shared, shared_vars.v_shared, test, add_updates)
            h = dropout(rectify(h_bn), p_drop_hidden)

            h_bn2 = batch_norm(T.dot(h, params.w_h2) + params.b_h2, params.gamma2, params.beta2, shared_vars.m_shared2, shared_vars.v_shared2, test, add_updates)
            h2 = dropout(rectify(h_bn2), p_drop_hidden)

            h_bn3 = batch_norm(T.dot(h2, params.w_h3) + params.b_h3, params.gamma3, params.beta3, shared_vars.m_shared3, shared_vars.v_shared3, test, add_updates)
            h3 = dropout(rectify(h_bn3), p_drop_hidden)

            py_x = softmax(T.dot(h3, params.w_o))
            return py_x
        
        add_updates = []
        noise_py_x = model(self.X, self.params, self.shared_vars, 0.2, 0.5, False, add_updates)
        cost = T.sum(T.nnet.categorical_crossentropy(noise_py_x, self.Y))
        
        pyx = model(self.X, self.params, self.shared_vars, 0., 0., True, None)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx, add_updates)

