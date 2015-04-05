import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class LM_sgru(ModelLMBase):
    def __init__(self, data, hp):
        super(LM_sgru, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 256
        self.dropout = 0.5

        self.params = Parameters()
        n_tokens = self.data['n_tokens']
        n_h = self.n_h
        scale = hp.init_scale
        gates = 2
        tst = 1  #self.hp.seq_size-1

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w_emb = shared_normal((n_tokens, n_h), scale=scale)
                #w_o = shared_normal((n_h, n_tokens), scale=scale)
                
                W1 = shared_normal((gates*tst, n_h, n_h), scale=hp.init_scale)
                V1 = shared_normal((gates*tst, n_h, n_h), scale=hp.init_scale)
                W2 = shared_normal((gates*tst, n_h, n_h), scale=hp.init_scale)
                V2 = shared_normal((gates*tst, n_h, n_h), scale=hp.init_scale)
                b1 = shared_zeros((gates*tst, n_h,))
                b2 = shared_zeros((gates*tst, n_h,))

                b1_h = shared_zeros((n_h,))
                b2_h = shared_zeros((n_h,))
        
        def sgru(X, h, W, U, b, t):
            t = 0
            z_t = T.tanh(T.dot(X,W[t*2+0]) + T.dot(h,U[t*2+0]) + b[t*2+0])
            r_t = T.tanh(T.dot(X,W[t*2+1]) + T.dot(h,U[t*2+1]) + b[t*2+1])
            return z_t * r_t

        def model(x, p, p_dropout):
            input_size = x.shape[1]

            h0 = p.w_emb[x]  # (seq_len, batch_size, emb_size)
            h0 = dropout(h0, p_dropout)

            cost, h1, h2, = [0, batch_col(input_size, p.b1_h), batch_col(input_size, p.b2_h)]
                               
            for t in xrange(0, self.hp.seq_size-1):
                h1 = sgru(h0[t], h1, p.W1, p.V1, p.b1, t)
                h1 = dropout(h1, p_dropout)
                h2 = sgru(h1, h2, p.W2, p.V2, p.b2, t)
                h2 = dropout(h2, p_dropout)

                if t >= self.hp.warmup_size:
                    pyx = softmax(T.dot(h2, T.transpose(p.w_emb)))
                    cost += T.sum(T.nnet.categorical_crossentropy(pyx, theano_one_hot(x[t+1], n_tokens)))

            #outputs_info = [0,
            #               batch_col(input_size, p.b1_h), 
            #               batch_col(input_size, p.b1_c), 
            #               batch_col(input_size, p.b2_h), 
            #               batch_col(input_size, p.b2_c)]
                               
            #def stepFull(t, h0, cost, h1, c1, h2, c2):
            #    h1, c1 = lstm(h0, h1, c1, p.W1, p.V1, p.b1)
            #    h1 = dropout(h1, p_dropout)
            #    h2, c2 = lstm(h1, h2, c2, p.W2, p.V2, p.b2)
            #    h2 = dropout(h2, p_dropout)
            #    cost = T.sum(T.nnet.categorical_crossentropy(pyx, theano_one_hot(x[t+1], n_tokens)))
            #    return cost, h1, c1, h2, c2

            #[cost, h1, c1, h2, c2,], _ = theano.scan(stepFull, n_steps=self.n_t, sequences=[T.arange(self.n_t), h0], outputs_info=outputs_info)
            #cost = T.sum(cost)

            return cost
        
        cost = model(self.X, self.params, self.dropout)
        te_cost = model(self.X, self.params, 0.)

        self.compile(cost, te_cost)

