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


class Vae1(ModelULBase):
    def __init__(self, data, hp):
        super(Vae1, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 800
        self.n_z = 20
        self.n_t = 1

        self.gaussian = False
            
        self.params = Parameters()
        n_x = self.data['n_x']
        n_h = self.n_h
        n_z = self.n_z
        n_t = self.n_t
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W1 = shared_normal((n_x, n_h), scale=scale)
                W11 = shared_normal((n_h, n_h), scale=scale)
                W111 = shared_normal((n_h, n_h), scale=scale)
                W2 = shared_normal((n_h, n_z), scale=scale)
                W3 = shared_normal((n_h, n_z), scale=scale)
                W4 = shared_normal((n_h, n_h), scale=scale)
                W44 = shared_normal((n_h, n_h), scale=scale)
                W444 = shared_normal((n_z, n_h), scale=scale)
                W5 = shared_normal((n_h, n_x), scale=scale)
                b1 = shared_zeros((n_h,))
                b11 = shared_zeros((n_h,))
                b111 = shared_zeros((n_h,))
                b2 = shared_zeros((n_z,))
                b3 = shared_zeros((n_z,))
                b4 = shared_zeros((n_h,))
                b44 = shared_zeros((n_h,))
                b444 = shared_zeros((n_h,))
                b5 = shared_zeros((n_x,))
        
        def encoder(x, p):
            h_encoder = T.tanh(T.dot(x,p.W1) + p.b1)
            h_encoder2 = T.tanh(T.dot(h_encoder,p.W11) + p.b11)
            h_encoder3 = T.tanh(T.dot(h_encoder2,p.W111) + p.b111)
        
            mu_encoder = T.dot(h_encoder3,p.W2) + p.b2
            log_sigma_encoder = 0.5*(T.dot(h_encoder3,p.W3) + p.b3)
            log_qpz = -0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))
        
            eps = srnd.normal(mu_encoder.shape, dtype=theano.config.floatX) 
            z = mu_encoder + eps*T.exp(log_sigma_encoder)
            return z, log_qpz
    
        def decoder(z, p, x=None):
            h_decoder3 = T.tanh(T.dot(z,p.W444) + p.b444)
            h_decoder2 = T.tanh(T.dot(h_decoder3,p.W44) + p.b44)
            h_decoder = T.tanh(T.dot(h_decoder2,p.W4) + p.b4)
        
            if self.gaussian:
                pxz = T.tanh(T.dot(h_decoder,p.W5) + p.b5)  
            else:
                pxz = T.nnet.sigmoid(T.dot(h_decoder,p.W5) + p.b5)
                
            if not x is None:
                if self.gaussian:
                    log_sigma_decoder = 0
                    log_pxz = 0.5 * np.log(2 * np.pi) + log_sigma_decoder + 0.5 * T.sum(T.sqr(x - pxz))  
                else:
                    log_pxz = T.nnet.binary_crossentropy(pxz,x).sum()
                return pxz, log_pxz
            else:
                return pxz

        x = binomial(self.X)
        z, log_qpz = encoder(x, self.params)
        pxz, log_pxz = decoder(z, self.params, x)
        cost = log_pxz + log_qpz

        s_pxz = decoder(self.Z, self.params)
        a_pxz = T.zeros((self.n_t, s_pxz.shape[0], s_pxz.shape[1]))
        a_pxz = T.set_subtensor(a_pxz[0,:,:], s_pxz)

        self.compile(log_pxz, log_qpz, cost, a_pxz)
