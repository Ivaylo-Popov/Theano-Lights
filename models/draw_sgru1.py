import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import scipy.io
import time
import sys
import logging
import copy

from toolbox import *
from modelbase import *


class Draw_sgru1(ModelULBase):
    def __init__(self, data, hp):
        super(Draw_sgru1, self).__init__(self.__class__.__name__, data, hp)
        
        n_x = self.data['n_x']
        n_h = 256
        n_t = 12
        n_zpt = 32
        gates = 4
        self.n_t = n_t
        self.n_z = n_t * n_zpt
        self.sample_steps = False

        self.params = Parameters()
        
        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W1 = shared_normal((gates*1, n_x + n_h, n_h), scale=hp.init_scale)
                W11 = shared_normal((gates*1, n_h, n_h), scale=hp.init_scale)
                W4 = shared_normal((gates*1, n_zpt, n_h), scale=hp.init_scale)
                W44 = shared_normal((gates*1, n_h, n_h), scale=hp.init_scale)
                b1 = shared_zeros((gates*1, n_h,))
                b4 = shared_zeros((gates*1, n_h,))
                b10_h = shared_zeros((n_h,))
                b40_h = shared_zeros((n_h,))
    
                W2 = shared_normal((n_h, n_zpt), scale=hp.init_scale)
                W3 = shared_normal((n_h, n_zpt), scale=hp.init_scale)
                W5 = shared_normal((n_h, n_x), scale=hp.init_scale)
                b2 = shared_zeros((n_zpt,))
                b3 = shared_zeros((n_zpt,))
                b5 = shared_zeros((n_x,))
                ex0 = shared_zeros((n_x,))
        
        def rnn(X, h, W, U, b, t):
            return T.tanh(T.dot(X,W[0]) + T.dot(h,U[0]) + b[0])

        #def gru(X, h, W, U, b):
        #    z_t = T.nnet.sigmoid(T.dot(X,W[:,:n_h]) + T.dot(h,U[:,:n_h]) + b[n_h])
        #    r_t = T.nnet.sigmoid(T.dot(X,W[:,n_h:2*n_h]) + T.dot(h,U[:,n_h:2*n_h]) + b[n_h:2*n_h])
        #    h_t = T.tanh(T.dot(X,W[:,2*n_h:3*n_h]) + r_t * T.dot(h,U[:,2*n_h:3*n_h]) + b[2*n_h:3*n_h])
        #    return (1 - z_t) * h + z_t * h_t

        def sgru(X, h, W, U, b, t):
            t = 0
            z_t = T.tanh(T.dot(X,W[t*2+0]) + T.dot(h,U[t*2+0]) + b[t*2+0])
            r_t = T.tanh(T.dot(X,W[t*2+1]) + T.dot(h,U[t*2+1]) + b[t*2+1])
            return z_t * r_t

        def sgru2(X, h, W, U, b, t):
            t = 0
            z_t = T.tanh(T.dot(X,W[t*2+0]) + b[t*2+0])
            r_t = (T.dot(h,U[t*2+0]) + b[t*2+1])
            return T.tanh(T.dot(z_t*r_t,T.transpose(U[t*2+1]))) 

        def sgru3(X, h, W, U, b, t):
            t = 0
            z_t = T.tanh(T.dot(X,W[t*2+0]) + b[t*2+0])
            r_t = (T.dot(h,U[t*2+0]) + b[t*2+1])
            z_t2 = (T.dot(X,W[t*2+1]) + b[t*2+2])
            r_t2 = T.tanh(T.dot(h,U[t*2+1]) + b[t*2+3])
            return T.tanh(T.dot(z_t*r_t,T.transpose(U[t*2+2])) + T.dot(z_t2*r_t2,T.transpose(U[t*2+3]))) 

        frnn = sgru

        # Encoder
        p = self.params

        x = self.X  # binomial(self.X)
        input_size = x.shape[0]
        ex = p.ex0
        h_encoder_h = p.b10_h
        h_decoder_h = T.zeros((input_size, p.b40_h.shape[0])) + p.b40_h
        log_qpz = 0

        for t in xrange(0, n_t):
            x_e = x - T.nnet.sigmoid(ex)
            h_x = concatenate([x_e, h_decoder_h], axis=1)
            h_encoder_h = frnn(h_x, h_encoder_h, p.W1, p.W11, p.b1, t)
            mu_encoder_t = T.dot(h_encoder_h, p.W2) + p.b2
            log_sigma_encoder_t = 0.5*(T.dot(h_encoder_h, p.W3) + p.b3) 
            log_qpz += -0.5* T.sum(1 + 2*log_sigma_encoder_t - mu_encoder_t**2 - T.exp(2*log_sigma_encoder_t))
        
            eps = srnd.normal(mu_encoder_t.shape, dtype=theano.config.floatX) 
            z = mu_encoder_t + eps*T.exp(log_sigma_encoder_t)
            h_decoder_h = frnn(z, h_decoder_h, p.W4, p.W44, p.b4, t)
            ex += T.dot(h_decoder_h, p.W5) + p.b5
            
        pxz = T.nnet.sigmoid(ex)
        log_pxz = T.nnet.binary_crossentropy(pxz, x).sum()
        cost = log_pxz + log_qpz
    
        # Generate
        z = self.Z.reshape((-1, n_t, n_zpt), ndim=3)
        input_size = z.shape[0]
        s_ex = p.ex0
        s_h_decoder_h = p.b40_h
        
        if self.sample_steps:
            a_pxz = T.zeros((n_t + 1, input_size, n_x))
        else:
            a_pxz = T.zeros((1, input_size, n_x))

        for t in xrange(0, n_t):
            if self.sample_steps:
                a_pxz = T.set_subtensor(a_pxz[t,:,:], T.nnet.sigmoid(s_ex))
            s_h_decoder_h = frnn(z[:,t,:], s_h_decoder_h, p.W4, p.W44, p.b4, t)
            s_ex +=  T.dot(s_h_decoder_h, p.W5) + p.b5
            
        a_pxz = T.set_subtensor(a_pxz[-1,:,:], T.nnet.sigmoid(s_ex))

        self.compile(log_pxz, log_qpz, cost, a_pxz)
