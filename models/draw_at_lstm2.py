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
import cPickle as pickle
import copy

from toolbox import *
from modelbase import *


class Draw_at_lstm2(ModelULBase):
    def __init__(self, data, hp):
        super(Draw_at_lstm2, self).__init__(self.__class__.__name__, data, hp)
        
        self.sample_steps = True
        self.n_h = 256
        self.n_t = 12
        self.n_zpt = 100
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

        # Attention
        read_n = 5
        self.zoomer = AttentionDraw(self.data['shape_x'][0], self.data['shape_x'][1], read_n)

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                AR = shared_normal((n_h, self.zoomer.n_att_params), scale=scale)
                AW_l = shared_normal((n_h, self.zoomer.n_att_params), scale=scale)
                AW_w = shared_normal((n_h, read_n**2), scale=scale)
                baw_l = shared_zeros((self.zoomer.n_att_params,))
                baw_w = shared_zeros((read_n**2,))

                W1 = shared_normal((read_n**2 + n_h, n_h*gates), scale=scale)
                W11 = shared_normal((n_h, n_h*gates), scale=scale)
                W4 = shared_normal((n_zpt, n_h*gates), scale=scale)
                W44 = shared_normal((n_h, n_h*gates), scale=scale)
                b1 = shared_zeros((n_h*gates,))
                b4 = shared_zeros((n_h*gates,))
                b10_h = shared_zeros((n_h,))
                b40_h = shared_zeros((n_h,))
                b10_c = shared_zeros((n_h,))
                b40_c = shared_zeros((n_h,))
    
                W2 = shared_normal((n_h, n_zpt), scale=scale)
                W3 = shared_normal((n_h, n_zpt), scale=scale)
                b2 = shared_zeros((n_zpt,))
                b3 = shared_zeros((n_zpt,))
                ex0 = shared_zeros((n_x,))
        
        def lstm(X, h, c, W, U, b, t):
            g_on = T.dot(X,W) + T.dot(h,U) + b
            i_on = T.nnet.sigmoid(g_on[:,:n_h])
            f_on = T.nnet.sigmoid(g_on[:,n_h:2*n_h])
            o_on = T.nnet.sigmoid(g_on[:,2*n_h:3*n_h])
            c = f_on * c + i_on * T.tanh(g_on[:,3*n_h:])
            h = o_on * T.tanh(c)
            return h, c 

        def attreader(x, x_e, h_decoder, t, p):
            l = T.dot(h_decoder, p.AR)
            rx_e   = self.zoomer.read(x_e, l)
            return T.concatenate([rx_e, h_decoder], axis=1)

        def attwriter(h_decoder, t, p):
            w = T.dot(h_decoder, p.AW_w) + p.baw_w
            l = T.dot(h_decoder, p.AW_l) + p.baw_l
            c_update = self.zoomer.write(w, l)
            return c_update

        # Encoder
        p = self.params
        frnn = lstm
        reader = attreader
        writer = attwriter

        x = binomial(self.X)
        input_size = x.shape[0]

        ex, log_qpz, h_encoder, h_decoder, c_encoder, c_decoder = [T.zeros((input_size, p.ex0.shape[0])) + p.ex0, 
                        0.0,
                        T.zeros((input_size, p.b10_h.shape[0])) + p.b10_h, 
                        T.zeros((input_size, p.b40_h.shape[0])) + p.b40_h,
                        T.zeros((input_size, p.b10_c.shape[0])) + p.b10_c, 
                        T.zeros((input_size, p.b40_c.shape[0])) + p.b40_c]

        for t in xrange(0, n_t):
            x_e = x - T.nnet.sigmoid(ex)  
            r_x = reader(x, x_e, h_decoder, t, p)
            h_encoder, c_encoder = frnn(r_x, h_encoder, c_encoder, p.W1, p.W11, p.b1, t)
            
            mu_encoder_t = T.dot(h_encoder, p.W2) + p.b2
            log_sigma_encoder_t = 0.5*(T.dot(h_encoder, p.W3) + p.b3) 
            log_qpz += -0.5* T.sum(1 + 2*log_sigma_encoder_t - mu_encoder_t**2 - T.exp(2*log_sigma_encoder_t))
            eps = srnd.normal((input_size, n_zpt), dtype=theano.config.floatX) 
            z = mu_encoder_t + eps*T.exp(log_sigma_encoder_t)
            
            h_decoder, c_decoder = frnn(z, h_decoder, c_decoder, p.W4, p.W44, p.b4, t)
            ex += writer(h_decoder, t, p)

        pxz = T.nnet.sigmoid(ex)

        log_pxz = T.nnet.binary_crossentropy(pxz, x).sum()
        cost = log_pxz + log_qpz
    
        # Generate
        z = self.Z.reshape((-1, n_t, n_zpt), ndim=3)
        input_size = z.shape[0]
        
        s_ex, s_h_decoder_h, s_h_decoder_c = [T.zeros((input_size, p.ex0.shape[0])) + p.ex0, 
                        T.zeros((input_size, p.b40_h.shape[0])) + p.b40_h,
                        T.zeros((input_size, p.b40_c.shape[0])) + p.b40_c]

        if self.sample_steps:
            a_pxz = T.zeros((n_t + 1, input_size, n_x))
        else:
            a_pxz = T.zeros((1, input_size, n_x))
        
        for t in xrange(0, n_t):
            s_h_decoder_h, s_h_decoder_c = frnn(z[:,t,:], s_h_decoder_h, s_h_decoder_c, p.W4, p.W44, p.b4, t)
            s_ex += writer(s_h_decoder_h, t, p)

            if self.sample_steps:
                a_pxz = T.set_subtensor(a_pxz[t,:,:], T.nnet.sigmoid(s_ex))

        a_pxz = T.set_subtensor(a_pxz[-1,:,:], T.nnet.sigmoid(s_ex))

        self.compile(log_pxz, log_qpz, cost, a_pxz)
