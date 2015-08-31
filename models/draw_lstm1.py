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


class Draw_lstm1(ModelULBase):
    def __init__(self, data, hp):
        super(Draw_lstm1, self).__init__(self.__class__.__name__, data, hp)
        
        self.sample_steps = True
        self.n_h = 256
        self.n_t = 12
        self.n_zpt = 32
        self.n_z = self.n_t * self.n_zpt
        self.gates = 4

        self.params = Parameters()
        n_x = self.data['n_x']
        n_h = self.n_h
        n_z = self.n_z
        n_zpt = self.n_zpt
        n_t = self.n_t
        gates = self.gates
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W1 = shared_normal((n_x*2 + n_h, n_h*gates), scale=scale)
                W11 = shared_normal((n_h, n_h*gates), scale=scale)
                W2 = shared_normal((n_h, n_zpt), scale=scale)
                W3 = shared_normal((n_h, n_zpt), scale=scale)
                W4 = shared_normal((n_zpt, n_h*gates), scale=scale)
                W44 = shared_normal((n_h, n_h*gates), scale=scale)
                W5 = shared_normal((n_h, n_x), scale=scale)
                b1 = shared_zeros((n_h*gates,))
                b10_h = shared_zeros((n_h,))
                b10_c = shared_zeros((n_h,))
                b2 = shared_zeros((n_zpt,))
                b3 = shared_zeros((n_zpt,))
                b4 = shared_zeros((n_h*gates,))
                b40_h = shared_zeros((n_h,))
                b40_c = shared_zeros((n_h,))
                b5 = shared_zeros((n_x,))
                c0 = shared_zeros((n_x,))
        
        def lstm(X, h, c, W, U, b):
            g_on = T.dot(X,W) + T.dot(h,U) + b
            i_on = T.nnet.sigmoid(g_on[:,:n_h])
            f_on = T.nnet.sigmoid(g_on[:,n_h:2*n_h])
            o_on = T.nnet.sigmoid(g_on[:,2*n_h:3*n_h])

            c = f_on * c + i_on * T.tanh(g_on[:,3*n_h:])
            h = o_on * T.tanh(c)
            return h, c 

        # Encoder
        p = self.params

        x = self.X # binomial(self.X)
        input_size = x.shape[0]

        c = p.c0
        h_encoder_h = p.b10_h
        h_encoder_c = p.b10_c
        h_decoder_h = T.zeros((input_size,p.b40_h.shape[0])) + p.b40_h  
        h_decoder_c = p.b40_c
        log_qpz = 0

        eps = srnd.normal((n_t, input_size, n_zpt), dtype=theano.config.floatX) 

        for t in xrange(0, n_t):
            x_e = x - T.nnet.sigmoid(c)
            h_x = concatenate([x, x_e, h_decoder_h], axis=1)
            h_encoder_h, h_encoder_c = lstm(h_x, h_encoder_h, h_encoder_c, p.W1, p.W11, p.b1)
            mu_encoder = T.dot(h_encoder_h, p.W2) + p.b2
            log_sigma_encoder = 0.5*(T.dot(h_encoder_h, p.W3) + p.b3) 
            log_qpz += -0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))
            
            z = mu_encoder + eps[t]*T.exp(log_sigma_encoder)
            h_decoder_h, h_decoder_c = lstm(z, h_decoder_h, h_decoder_c, p.W4, p.W44, p.b4)
            c += T.dot(h_decoder_h, p.W5) + p.b5
            
        pxz = T.nnet.sigmoid(c)
        log_pxz = T.nnet.binary_crossentropy(pxz,x).sum()
        cost = log_pxz + log_qpz

        #out_log_qpz = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder), axis = 1)
        #out_log_pxz = -T.nnet.binary_crossentropy(pxz,x).sum(axis = 1)

        # Generate
        z = self.Z.reshape((-1, n_t, n_zpt), ndim=3)
        s_ex = p.c0
        s_h_decoder_h = p.b40_h
        s_h_decoder_c = p.b40_c
        
        if self.sample_steps:
            a_pxz = T.zeros((n_t + 1, z.shape[0], n_x))
        else:
            a_pxz = T.zeros((1, z.shape[0], n_x))

        for t in xrange(0, n_t):
            if self.sample_steps:
                a_pxz = T.set_subtensor(a_pxz[t,:,:], T.nnet.sigmoid(s_ex))
            s_h_decoder_h, s_h_decoder_c = lstm(z[:,t,:], s_h_decoder_h, s_h_decoder_c, p.W4, p.W44, p.b4)
            s_ex += T.dot(s_h_decoder_h, p.W5) + p.b5
            
        a_pxz = T.set_subtensor(a_pxz[-1,:,:], T.nnet.sigmoid(s_ex))

        self.compile(log_pxz, log_qpz, cost, a_pxz)

