import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class LM_draw(ModelLMBase):
    def __init__(self, data, hp):
        super(LM_draw, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 1024
        self.n_zpt = 256
        self.dropout = 0.0

        self.params = Parameters()
        self.hiddenstates = Parameters()
        n_tokens = self.data['n_tokens']
        gates = 4

        with self.hiddenstates:
            b1_h = shared_zeros((self.hp.batch_size, self.n_h))
            b1_c = shared_zeros((self.hp.batch_size, self.n_h))
            b2_h = shared_zeros((self.hp.batch_size, self.n_h))
            b2_c = shared_zeros((self.hp.batch_size, self.n_h))

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W_emb = shared_normal((n_tokens, self.n_h), scale=hp.init_scale)
                
                W1 = shared_normal((self.n_h*2, self.n_h*gates), scale=hp.init_scale*1.5)
                V1 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                b1 = shared_zeros((self.n_h*gates,))
                
                Wmu = shared_normal((self.n_h, self.n_zpt), scale=hp.init_scale)
                Wsi = shared_normal((self.n_h, self.n_zpt), scale=hp.init_scale)
                bmu = shared_zeros((self.n_zpt,))
                bsi = shared_zeros((self.n_zpt,))

                W2 = shared_normal((self.n_zpt, self.n_h*gates), scale=hp.init_scale*1.5)
                V2 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                b2 = shared_zeros((self.n_h*gates,))
        
        def lstm(X, h, c, W, U, b):
            g_on = T.dot(X,W) + T.dot(h,U) + b
            i_on = T.nnet.sigmoid(g_on[:,:self.n_h])
            f_on = T.nnet.sigmoid(g_on[:,self.n_h:2*self.n_h])
            o_on = T.nnet.sigmoid(g_on[:,2*self.n_h:3*self.n_h])
            c = f_on * c + i_on * T.tanh(g_on[:,3*self.n_h:])
            h = o_on * T.tanh(c)
            return h, c

        def model(x, p, p_dropout, noise):
            input_size = x.shape[1]

            h0 = p.W_emb[x]  # (seq_len, batch_size, emb_size)
            h0 = dropout(h0, p_dropout)

            cost, h1, c1, h2, c2 = [0., b1_h, b1_c, b2_h, b2_c]
            eps = srnd.normal((self.hp.seq_size, input_size, self.n_zpt), dtype=theano.config.floatX)
            
            for t in xrange(0, self.hp.seq_size):
                if t >= self.hp.warmup_size:
                    pyx = softmax(T.dot(h2, T.transpose(p.W_emb)))
                    cost += T.sum(T.nnet.categorical_crossentropy(pyx, theano_one_hot(x[t], n_tokens)))

                h_x = concatenate([h0[t], h2], axis=1)
                h1, c1 = lstm(h_x, h1, c1, p.W1, p.V1, p.b1)
                h1 = dropout(h1, p_dropout)

                mu_encoder = T.dot(h1, p.Wmu) + p.bmu
                if noise:
                    log_sigma_encoder = 0.5*(T.dot(h1, p.Wsi) + p.bsi) 
                    cost += -0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder)) * 0.01
                    z = mu_encoder + eps[t]*T.exp(log_sigma_encoder)
                else:
                    z = mu_encoder

                h2, c2 = lstm(z, h2, c2, p.W2, p.V2, p.b2)
                h2 = dropout(h2, p_dropout)

            h_updates = [(b1_h, h1), (b1_c, c1), (b2_h, h2), (b2_c, c2)]
            return cost, h_updates
        
        cost, h_updates = model(self.X, self.params, self.dropout, True)
        te_cost, te_h_updates = model(self.X, self.params, 0.0, False)

         
        def generate(seed_idx, p):
            spx = T.zeros((self.hp.seq_size, n_tokens))
            h1, c1, h2, c2 = [T.zeros((self.n_h)), 
                              T.zeros((self.n_h)), 
                              T.zeros((self.n_h)), 
                              T.zeros((self.n_h))]

            spx = T.set_subtensor(spx[0, seed_idx], 1)
                            
            #for t in xrange(0, self.hp.seq_size):
            #    if t > 0:
            #        pyx = softmax(T.dot(h2, T.transpose(p.W_emb)))
            #        spx = T.set_subtensor(spx[t,:], srnd.multinomial(pvals=pyx)[0])

            #    h1, c1 = lstm(p.W_emb[T.cast(spx[t], dtype='int32')], h1, c1, p.W1, p.V1, p.b1)
            #    h2, c2 = lstm(h1, h2, c2, p.W2, p.V2, p.b2)

            return spx

        spx = generate(self.seed_idx, self.params)

        self.compile(cost, te_cost, h_updates, te_h_updates, spx)

