import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class RBFN(ModelSLBase):
    def __init__(self, data, hp):
        """
        Radias basis function feedforward neural network
        """

        super(RBFN, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 800
        n_c = 100

        self.params = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h = self.n_h
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h), scale=scale)
                b_h = shared_normal((n_h,), scale=0)
                w_h2 = shared_normal((n_c, n_h), scale=scale)
                b_h2 = shared_normal((n_h,), scale=0)
                w_h3 = shared_normal((n_h, n_h), scale=scale)
                b_h3 = shared_normal((n_h,), scale=0)
                w_o = shared_normal((n_h, n_y), scale=scale)

                c = shared_normal((n_c, n_h), scale=scale)
                s = shared_normal((n_c,), scale=scale)
        
        def model(X, params, p_drop_input, p_drop_hidden):
            X = dropout(X, p_drop_input)
            h = dropout(rectify(T.dot(X, params.w_h) + params.b_h ), p_drop_hidden)

            C = params.c[np.newaxis, :, :]  # 0, n_c, n_x
            X_rbf = h[:, np.newaxis, :]  # batch_size, 0, n_x
            difnorm = T.sum((C-X_rbf)**2, axis=-1)
            h_rbf = T.exp(-difnorm * (params.s**2))

            h2 = dropout(rectify(T.dot(h_rbf, params.w_h2) + params.b_h2), p_drop_hidden)
            h3 = dropout(rectify(T.dot(h2, params.w_h3) + params.b_h3), p_drop_hidden)

            py_x = softmax(T.dot(h3, params.w_o))
            return py_x
        
        noise_py_x = model(self.X, self.params, 0.2, 0.5)
        cost = T.sum(T.nnet.categorical_crossentropy(noise_py_x, self.Y))
        
        pyx = model(self.X, self.params, 0., 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, T.argmax(self.Y, axis=1)))

        self.compile(cost, error_map_pyx)

