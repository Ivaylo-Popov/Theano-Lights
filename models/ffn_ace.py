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


class FFN_ace(ModelSLBase):
    """
    Auto-classifier-encoder (Georgiev, 2015)
    """

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)

    def __init__(self, data, hp):
        super(FFN_ace, self).__init__(self.__class__.__name__, data, hp)
        
        # batch_size: 10000; learning_rate = 0.0015; lr_halflife = 200, 500

        self.epsilon = 0.0001

        self.params = Parameters()
        self.shared_vars = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h1 = 1200
        n_h2 = 1000
        n_h3 = 800
        n_h4 = 800
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h1), scale=scale)
                b_h = shared_zeros((n_h1,))
                w_h2 = shared_normal((n_h1, n_h2), scale=scale)
                b_h2 = shared_zeros((n_h2,))
                w_h3 = shared_normal((n_h2, n_h3), scale=scale)
                b_h3 = shared_zeros((n_h3,))
                w_h4 = shared_normal((n_h3, n_h4), scale=scale)
                b_h4 = shared_zeros((n_h4,))
                w_o = shared_normal((n_h4, n_y), scale=scale)
        
        def batch_norm(h):
	        m = T.mean(h, axis=0, keepdims=True)
	        std = T.sqrt(T.var(h, axis=0, keepdims=True) + self.epsilon)
	        h = (h - m) / std
	        return h

        def model(X, params, p_drop_input, p_drop_hidden):
	        X_noise = X + gaussian(X.shape, p_drop_input)
	        h = batch_norm(dropout(rectify(T.dot(X_noise, params.w_h) + params.b_h), p_drop_hidden))

	        # Dual reconstruction error
	        phx =  T.nnet.sigmoid(T.dot(h, T.dot(h.T, X_noise)) / self.hp.batch_size)
	        log_phx = T.nnet.binary_crossentropy(phx, X_noise).sum()
            
	        h2 = dropout(rectify(T.dot(h, params.w_h2) + params.b_h2), p_drop_hidden) 
	        
	        h3 = batch_norm(dropout(rectify(T.dot(h2, params.w_h3) + params.b_h3), p_drop_hidden))
	        h4 = dropout(rectify(T.dot(h3, params.w_h4) + params.b_h4), p_drop_hidden)
					
	        py_x = softmax(T.dot(h4, params.w_o))
	        return [py_x, log_phx]
        
        noise_py_x, cost_recon = model(self.X, self.params, 0.2, 0.5)
        cost_y2 = -T.sum(self.Y * T.log(noise_py_x))
        cost = cost_y2 + cost_recon
        
        pyx, _ = model(self.X, self.params, 0., 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx)

