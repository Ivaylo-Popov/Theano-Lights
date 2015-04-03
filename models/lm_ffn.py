import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class LM_ffn(ModelLMBase):
    def __init__(self, data, hp):
        super(LM_ffn, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_emb = 128
        self.n_h = 256

        self.params = Parameters()
        n_tokens = self.data['n_tokens']
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h = self.n_h
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w_emb = shared_normal((n_tokens, self.n_emb), scale=scale)
                
                w_h1 = shared_normal((self.n_emb*n_x, n_h), scale=scale)
                b_h1 = shared_normal((n_h,), scale=0)
                w_h2 = shared_normal((n_h, self.n_emb), scale=scale)
                b_h2 = shared_normal((self.n_emb,), scale=0)
        
        def model(x, params, p_drop_input, p_drop_hidden):
            
            shape = (x.shape[0], x.shape[1] * params.w_emb.shape[1])
            h_emb = params.w_emb[x.flatten()].reshape(shape)

            h1 = dropout(rectify(T.dot(h_emb, params.w_h1) + params.b_h1 ), p_drop_hidden)
            h2 = dropout(rectify(T.dot(h1, params.w_h2) + params.b_h2), p_drop_hidden)

            pyx = softmax(T.dot(h2, T.transpose(params.w_emb)))
            return pyx
        
        noise_pyx = model(self.X, self.params, 0.2, 0.2)

        cost = T.sum(T.nnet.categorical_crossentropy(noise_pyx, theano_one_hot(self.Y, n_tokens)))

        pyx = model(self.X, self.params, 0., 0.)
        map_pyx = T.argmax(pyx, axis=1)
        error_map_pyx = T.sum(T.neq(map_pyx, self.Y))

        self.compile(cost, error_map_pyx)

