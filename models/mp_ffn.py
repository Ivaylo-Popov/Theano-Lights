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
        
        self.n_h = 200
        self.dropout = 0.0
        
        self.hiddenstates = Parameters()
        self.params = Parameters()

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W0 = shared_normal((self.data['n_x']*(self.hp.seq_size-1), self.n_h), scale=hp.init_scale)
                b0 = shared_zeros((self.n_h,))

                W1 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b1 = shared_zeros((self.n_h,))

                # Gaussian P(y|x)
                W_mu = shared_normal((self.n_h, self.data['n_y']), scale=hp.init_scale)
                b_mu = shared_zeros((self.data['n_y'],))
                #W_sigma = shared_normal((self.n_h, self.data['n_x']), scale=hp.init_scale)
                #b_sigma = shared_zeros((self.data['n_x'],))
                
        
        def model(X, Y, p, p_dropout):
            x_in = X[:self.hp.seq_size-1].transpose(1,0,2)
            x_in = x_in.reshape((x_in.shape[0], -1))
            h0 = T.tanh(T.dot(x_in, p.W0) + p.b0)
            h1 = T.tanh(T.dot(h0,p.W1) + p.b1)
            mu_pyx = T.dot(h1, p.W_mu) + p.b_mu
                    
            cost = 0.5*T.sum((Y[self.hp.seq_size-1,:,10] - mu_pyx[:,10])**2)
            cost_base = 0.5*T.sum(Y[self.hp.seq_size-1,:,10]**2)
                
            return cost, cost_base
        
        cost, cost_base = model(self.X, self.Y, self.params, self.dropout)
        te_cost, te_cost_base = model(self.X, self.Y, self.params, 0.0)

        self.compile(cost, te_cost, [], [], cost_base, te_cost_base)

