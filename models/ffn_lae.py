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


class FFN_lae(ModelSLBase):
    """
    Auto-encoder with lateral connections
    """

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)

    def __init__(self, data, hp):
        super(FFN_lae, self).__init__(self.__class__.__name__, data, hp)
        
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
                w_h1 = shared_normal((n_x, n_h1), scale=scale)
                v_h1 = shared_normal((n_h1, n_x), scale=scale)
                u_h1 = shared_normal((n_h1, n_x), scale=scale)
                b_h1 = shared_zeros((n_h1,))
                c_h1 = shared_zeros((n_x,))
                w_h2 = shared_normal((n_h1, n_h2), scale=scale)
                v_h2 = shared_normal((n_h2, n_h1), scale=scale)
                u_h2 = shared_normal((n_h2, n_h1), scale=scale)
                b_h2 = shared_zeros((n_h2,))
                c_h2 = shared_zeros((n_h1,))
                w_h3 = shared_normal((n_h2, n_h3), scale=scale)
                v_h3 = shared_normal((n_h3, n_h2), scale=scale)
                u_h3 = shared_normal((n_h3, n_h2), scale=scale)
                b_h3 = shared_zeros((n_h3,))
                c_h3 = shared_zeros((n_h2,))
                w_h4 = shared_normal((n_h3, n_h4), scale=scale)
                u_h4 = shared_normal((n_h4, n_h3), scale=scale)
                b_h4 = shared_zeros((n_h4,))
                c_h4 = shared_zeros((n_h3,))
                w_o = shared_normal((n_h4, n_y), scale=scale)
        
        def batch_norm(h):
	        m = T.mean(h, axis=0, keepdims=True)
	        std = T.sqrt(T.var(h, axis=0, keepdims=True) + self.epsilon)
	        h = (h - m) / std
	        return h

        def model(X, p_drop_input, p_drop_hidden):
	        X_e = X + gaussian(X.shape, p_drop_input)
	        
            # Clean encoder
	        #h1 = rectify(batch_norm(T.dot(X, w_h1)) + b_h1)
	        #h2 = rectify(batch_norm(T.dot(h1, w_h2)) + b_h2)
	        #h3 = rectify(batch_norm(T.dot(h2, w_h3)) + b_h3)
	        #h4 = rectify(batch_norm(T.dot(h3, w_h4)) + b_h4)

            # Noisy encoder
	        h1_e = dropout(rectify(batch_norm(T.dot(X_e, w_h1)) + b_h1), p_drop_hidden)
	        h2_e = dropout(rectify(batch_norm(T.dot(h1_e, w_h2)) + b_h2), p_drop_hidden)
	        h3_e = dropout(rectify(batch_norm(T.dot(h2_e, w_h3)) + b_h3), p_drop_hidden)
	        h4_e = dropout(rectify(batch_norm(T.dot(h3_e, w_h4)) + b_h4), p_drop_hidden)

            # Decoder
	        d3 = rectify(T.dot(h4_e, u_h4) + c_h4)
	        d2 = rectify(T.dot(h3_e, u_h3) + T.dot(d3, v_h3) + c_h3)
	        d1 = rectify(T.dot(h2_e, u_h2) + T.dot(d2, v_h2) + c_h2)
	        dX = T.nnet.sigmoid(T.dot(h1_e, u_h1) + T.dot(d1, v_h1) + c_h1)

	        # Reconstruction error
	        log_pxh = 0.1*T.nnet.binary_crossentropy(dX, X).sum()
		
	        py_x = softmax(T.dot(h4_e, w_o))
	        return [py_x, log_pxh]
        
        noise_py_x, cost_recon = model(self.X, 0.2, 0.5)
        cost_y2 = -T.sum(self.Y * T.log(noise_py_x))
        cost = cost_y2 + cost_recon
        
        pyx, _ = model(self.X, 0., 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx)
