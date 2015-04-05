import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class LM_gru(ModelLMBase):
    def __init__(self, data, hp):
        super(LM_gru, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 256
        self.dropout = 0.5

        self.params = Parameters()
        self.hiddenstates = Parameters()
        n_tokens = self.data['n_tokens']
        n_h = self.n_h
        scale = hp.init_scale
        gates = 3

        with self.hiddenstates:
            b1_h = shared_zeros((self.hp.batch_size, n_h))
            b2_h = shared_zeros((self.hp.batch_size, n_h))

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W_emb = shared_normal((n_tokens, n_h), scale=scale)
                
                W1 = shared_normal((n_h, n_h*gates), scale=scale*1.5)
                V1 = shared_normal((n_h, n_h*gates), scale=scale*1.5)
                b1 = shared_zeros((n_h*gates))
                
                W2 = shared_normal((n_h, n_h*gates), scale=scale*1.5)
                V2 = shared_normal((n_h, n_h*gates), scale=scale*1.5)
                b2 = shared_zeros((n_h*gates,))

        
        def lstm(X, h, c, W, U, b):
            g_on = T.dot(X,W) + T.dot(h,U) + b
            i_on = T.nnet.sigmoid(g_on[:,:n_h])
            f_on = T.nnet.sigmoid(g_on[:,n_h:2*n_h])
            o_on = T.nnet.sigmoid(g_on[:,2*n_h:3*n_h])
            c = f_on * c + i_on * T.tanh(g_on[:,3*n_h:])
            h = o_on * T.tanh(c)
            return h, c

        def gru(X, h, W, U, b):
            z_t = T.nnet.sigmoid(T.dot(X,W[:,:n_h]) + T.dot(h,U[:,:n_h]) + b[:n_h])
            r_t = T.nnet.sigmoid(T.dot(X,W[:,n_h:2*n_h]) + T.dot(h,U[:,n_h:2*n_h]) + b[n_h:2*n_h])
            h_t = T.tanh(T.dot(X,W[:,2*n_h:3*n_h]) + r_t * T.dot(h,U[:,2*n_h:3*n_h]) + b[2*n_h:3*n_h])
            return (1 - z_t) * h + z_t * h_t

        def sgru(X, h, W, U, b):
            z_t = T.tanh(T.dot(X,W[:,:n_h]) + T.dot(h,U[:,:n_h]) + b[:n_h])
            h_t = T.tanh(T.dot(X,W[:,1*n_h:2*n_h]) + T.dot(h,U[:,1*n_h:2*n_h]) + b[1*n_h:2*n_h])
            return z_t * h_t

        def model(x, p, p_dropout):
            input_size = x.shape[1]

            h0 = p.W_emb[x]  # (seq_len, batch_size, emb_size)
            h0 = dropout(h0, p_dropout)

            cost, h1, h2 = [0., b1_h, b2_h]
                               
            for t in xrange(0, self.hp.seq_size):
                if t >= self.hp.warmup_size:
                    pyx = softmax(T.dot(dropout(h2, p_dropout), T.transpose(p.W_emb)))
                    cost += T.sum(T.nnet.categorical_crossentropy(pyx, theano_one_hot(x[t], n_tokens)))

                h1 = gru(h0[t], h1, p.W1, p.V1, p.b1)
                h2 = gru(dropout(h1, p_dropout), h2, p.W2, p.V2, p.b2)

            h_updates = [(b1_h, h1), (b2_h, h2)]

            return cost, h_updates
        
        cost, h_updates = model(self.X, self.params, self.dropout)
        te_cost, te_h_updates = model(self.X, self.params, 0.)

        self.compile(cost, te_cost, h_updates, te_h_updates)

