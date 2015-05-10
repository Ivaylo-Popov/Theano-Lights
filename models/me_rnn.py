import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class ME_rnn(ModelMPBase):
    def __init__(self, data, hp):
        super(ME_rnn, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 600
        self.dropout = 0.1
        self.delay = 0

        self.params = Parameters()
        self.hiddenstates = Parameters()

        self.trading_inputs = self.data['trading_inputs']

        with self.hiddenstates:
            b1_h = shared_zeros((self.hp.batch_size, self.n_h))
            b2_h = shared_zeros((self.hp.batch_size, self.n_h))

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W0 = shared_normal((self.data['n_x'], self.n_h), scale=hp.init_scale)
                b0 = shared_zeros((self.n_h,))

                W_mu = shared_normal((self.n_h, self.data['n_y']), scale=hp.init_scale)
                b_mu = shared_zeros((self.data['n_y'],))
                
                W1 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                V1 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b1 = shared_zeros((self.n_h,))
                
                W2 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                V2 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b2 = shared_zeros((self.n_h,))

                W3 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b3 = shared_zeros((self.n_h,))
        
        def rnn(X, h, W, U, b):
            return T.tanh(T.dot(X,W) + T.dot(h,U) + b)

        def model(X, Y, p, p_dropout):
            h0 = T.tanh(T.dot(X.reshape((-1, X.shape[-1])), p.W0) + p.b0)  
            h0 = dropout(h0, p_dropout)
            h0 = h0.reshape((-1, X.shape[1], h0.shape[-1]))

            cost, cost_var, m, info, h1, h2 = [0., 0., 0., 0., b1_h, b2_h]
            exposure = T.zeros_like(Y, dtype='float32')

            #self.trading_inputs[0] = self.trading_inputs[0] * 0.0    ###

            for t in xrange(0, self.hp.seq_size-self.delay):
                #if t % 2==0:
                if t >= self.hp.warmup_size:
                    h_decoder = T.tanh(T.dot(h2, p.W3) + p.b3)
                    m_new = T.tanh(T.dot(h_decoder, p.W_mu) + p.b_mu)*self.trading_inputs[1]

                    cost += -T.sum(Y[t]*m_new - abs(m_new-m)*self.trading_inputs[0])
                    cost_var += T.sqr(T.sum(Y[t]*m_new - abs(m_new-m)*self.trading_inputs[0]))

                    info += T.concatenate((
                        T.sum((Y[t]*m_new - abs(m_new-m)*self.trading_inputs[0])/self.trading_inputs[1], axis=0, keepdims=True),
                        T.sum(abs(m_new-m)/self.trading_inputs[1], axis=0, keepdims=True),
                        T.sum(abs(m_new)/self.trading_inputs[1], axis=0, keepdims=True)), axis=0).transpose()

                    exposure = T.set_subtensor(exposure[t], m_new/self.trading_inputs[1])
                    m = m_new

                h1 = rnn(h0[t], h1, p.W1, p.V1, p.b1)
                h1 = dropout(h1, p_dropout)
                h2 = rnn(h1, h2, p.W2, p.V2, p.b2)
                h2 = dropout(h2, p_dropout)
            
            # Final transaction costs
            cost += T.sum(abs(m)*self.trading_inputs[0])

            #cost_grad = cost / T.sqrt(cost_var)
            cost_grad = cost

            exposure = T.concatenate((exposure, Y), axis=2)

            h_updates = [(b1_h, h1), (b2_h, h2)]
            return cost, h_updates, info, cost_grad, cost_var, exposure
        
        cost, h_updates, info, cost_grad, cost_var, _ = model(self.X, self.Y[self.delay:], self.params, self.dropout)
        te_cost, te_h_updates, te_info, _, te_cost_var, exposure = model(self.X, self.Y[self.delay:], self.params, 0.0)

        self.compile(cost_grad, cost, te_cost, h_updates, te_h_updates, cost_var, te_cost_var, info, te_info, exposure=exposure)

