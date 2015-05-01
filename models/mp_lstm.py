import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class MP_lstm(ModelMPBase):
    def __init__(self, data, hp):
        super(MP_lstm, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 20
        self.dropout = 0.0
        
        self.params = Parameters()
        self.hiddenstates = Parameters()
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
                W0 = shared_normal((self.data['n_x'], self.n_h), scale=hp.init_scale)
                b0 = shared_zeros((self.n_h,))

                #W_m = shared_normal((self.n_h, self.data['n_x']), scale=hp.init_scale)
                #W_s = shared_normal((self.n_h, self.data['n_x']), scale=hp.init_scale)

                # Gaussian P(y|x)
                W_mu = shared_normal((self.n_h, self.data['n_x']), scale=hp.init_scale)
                b_mu = shared_zeros((self.data['n_x'],))
                #W_sigma = shared_normal((self.n_h, self.data['n_x']), scale=hp.init_scale)
                #b_sigma = shared_zeros((self.data['n_x'],))
                
                W1 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                V1 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                b1 = shared_zeros((self.n_h*gates,))
                
                W2 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                V2 = shared_normal((self.n_h, self.n_h*gates), scale=hp.init_scale*1.5)
                b2 = shared_zeros((self.n_h*gates,))

                W3 = shared_normal((self.n_h, self.n_h), scale=hp.init_scale)
                b3 = shared_zeros((self.n_h,))
        
        def rnn(X, h, c, W, U, b):
            g_on = T.tanh(T.dot(X,W) + T.dot(h,U) + b)
            h = g_on[:,:self.n_h]
            c = g_on[:,self.n_h:2*self.n_h]
            return h, c

        def lstm(X, h, c, W, U, b):
            g_on = T.dot(X,W) + T.dot(h,U) + b
            i_on = T.nnet.sigmoid(g_on[:,:self.n_h])
            f_on = T.nnet.sigmoid(g_on[:,self.n_h:2*self.n_h])
            o_on = T.nnet.sigmoid(g_on[:,2*self.n_h:3*self.n_h])
            c = f_on * c + i_on * T.tanh(g_on[:,3*self.n_h:])
            h = o_on * T.tanh(c)
            return h, c

        def model(x, p, p_dropout):
            x = x * 500
            batch_size = x.shape[1]
            h0 = T.tanh(T.dot(x.reshape((-1, x.shape[-1])), p.W0) + p.b0)  # (seq_len * batch_size, features_size)
            h0 = dropout(h0, p_dropout)
            h0 = h0.reshape((-1, batch_size, h0.shape[-1]))

            cost, cost_mse, h1, c1, h2, c2 = [0., 0., b1_h, b1_c, b2_h, b2_c]
                               
            for t in xrange(0, self.hp.seq_size):
                if t >= self.hp.warmup_size:
                    h_decoder = T.tanh(T.dot(h2,W3) + b3)
                    mu_pyx = T.dot(h_decoder, p.W_mu) + p.b_mu
                    #log_sigma_pyx = 0.5*(T.dot(h2, p.W_sigma) + p.b_sigma)
                    #cost += T.sum(0.5*np.log(2*np.pi) + log_sigma_pyx + 0.5*((x[t] - x[t-1] - mu_pyx)**2 / T.exp(2*log_sigma_pyx)))
                    #cost += T.sum(0.5*(x[t] - mu_pyx)**2)
                    #cost_mse += T.sum(0.5*(x[t] - x[t-1] - mu_pyx)**2)
                    
                    cost += T.sum(0.5*(x[t] - x[t-1] - mu_pyx)**2)
                    cost_mse += T.sum(0.5*(x[t] - x[t-1])**2)
                
                #h = (x[t] - T.dot(h1, W_m)) * T.dot(h1, W_s)

                h1, c1 = lstm(h0[t], h1, c1, p.W1, p.V1, p.b1)
                h1 = dropout(h1, p_dropout)
                h2, c2 = lstm(h1, h2, c2, p.W2, p.V2, p.b2)
                h2 = dropout(h2, p_dropout)

            h_updates = [(b1_h, h1), (b1_c, c1), (b2_h, h2), (b2_c, c2)]
            return cost, cost_mse, h_updates
        
        cost, cost_mse, h_updates = model(self.X, self.params, self.dropout)
        te_cost, te_cost_mse, te_h_updates = model(self.X, self.params, 0.0)

        self.compile(cost, te_cost, h_updates, te_h_updates, cost_mse, te_cost_mse)

