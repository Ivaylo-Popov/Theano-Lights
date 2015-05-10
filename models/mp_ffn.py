import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class MP_ffn(ModelMPBase):
    def __init__(self, data, hp):
        super(MP_ffn, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 500
        self.dropout = 0.0
        
        self.hiddenstates = Parameters()
        self.params = Parameters()
        self.delay = 1
        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W0 = shared_normal((self.data['n_x']*(self.hp.seq_size-1-self.delay), self.n_h), scale=hp.init_scale)
                b0 = shared_zeros((self.n_h,))

                W1 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b1 = shared_zeros((self.n_h,))
                W2 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b2 = shared_zeros((self.n_h,))

                W_mu = shared_normal((self.n_h, self.data['n_y']), scale=hp.init_scale)
                b_mu = shared_zeros((self.data['n_y'],))
        
        def model(X, Y, p, p_dropout):
            x_in = X[:self.hp.seq_size-1-self.delay].transpose(1,0,2)
            x_in = x_in.reshape((x_in.shape[0], -1))

            h0 = dropout(T.tanh(T.dot(x_in, p.W0) + p.b0), p_dropout)
            h1 = dropout(T.tanh(T.dot(h0,p.W1) + p.b1), p_dropout)
            h2 = dropout(T.tanh(T.dot(h1,p.W2) + p.b2), p_dropout)
            mu_pyx = T.dot(h2, p.W_mu) + p.b_mu
                    
            #cost = 0.5*T.sum((Y[self.hp.seq_size-1,:,4] - mu_pyx[:,4])**2)
            #cost_base = 0.5*T.sum(Y[self.hp.seq_size-1,:,4]**2)

            cost = 0.5*T.sum((Y[self.hp.seq_size-1] - mu_pyx)**2)
            cost_base = 0.5*T.sum(Y[self.hp.seq_size-1]**2)

            info = T.concatenate((0.5*T.sum((Y[self.hp.seq_size-1] - mu_pyx)**2, axis=0, keepdims=True), 
                                  0.5*T.sum(Y[self.hp.seq_size-1]**2, axis=0, keepdims=True)), axis=0).transpose()
                
            return cost, cost_base, info
        
        mult_Y = 2.
        cost, cost_base, info = model(self.X, self.Y*mult_Y, self.params, self.dropout)
        te_cost, te_cost_base, te_info = model(self.X, self.Y*mult_Y, self.params, 0.0)

        self.compile(cost, cost, te_cost, [], [], cost_base, te_cost_base, info, te_info)

