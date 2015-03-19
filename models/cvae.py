import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *


class Cvae(ModelULBase):
    def __init__(self, data, hp):
        super(Cvae, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 512
        self.n_z = 20
        self.n_t = 1

        self.gaussian = False
            
        self.params = Parameters()
        n_x = self.data['n_x']
        n_h = self.n_h
        n_z = self.n_z
        n_t = self.n_t
        scale = hp.init_scale
        downpool_sz = 28 // 4   # assume square 2D, 2 layers mean downsampling by 2 ** 2 -> 4

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            l1_e = (32, 1, 3, 3)  
            l2_e = (64, l1_e[0], 3, 3)  
            l3_e = (l2_e[0] * downpool_sz * downpool_sz, n_h)
            lz_e = (n_h, n_z)
            l1_d = (l1_e[1], l1_e[0], l1_e[2], l1_e[3])
            l2_d = (l2_e[1], l2_e[0], l2_e[2], l2_e[3])
            l3_d = (l3_e[1], l3_e[0])
            lz_d = (lz_e[1], lz_e[0])

            with self.params:
                w1e = shared_normal(l1_e, scale=scale)
                w2e = shared_normal(l2_e, scale=scale)
                w3e = shared_normal(l3_e, scale=scale)
                b3e = shared_zeros(l3_e[1])
                wmu = shared_normal(lz_e, scale=scale)
                bmu = shared_zeros(n_z)
                wsigma = shared_normal(lz_e, scale=scale)
                bsigma = shared_zeros(n_z)

                wd = shared_normal(lz_d, scale=scale)
                bd = shared_zeros((lz_d[1]))
                w3d = shared_normal(l3_d, scale=scale)
                b3d = shared_zeros((l3_d[1]))
                w2d = shared_normal(l2_d, scale=scale)
                w1d = shared_normal(l1_d, scale=scale)
                b1d = shared_normal((1, 28, 28))
        
        def conv_enc(X, p):
            h1 = rectify(max_pool_2d(conv(X, p.w1e), (2, 2)))
            h2 = rectify(max_pool_2d(conv(h1, p.w2e), (2, 2)))
            h2 = T.flatten(h2, outdim=2)
            h3 = T.tanh(T.dot(h2, p.w3e) + p.b3e)
            mu = T.dot(h3, p.wmu) + p.bmu
            log_sigma = 0.5 * (T.dot(h3, p.wsigma) + p.bsigma)
            return mu, log_sigma

        def deconv_dec(Z, p):
            h3 = T.tanh(T.dot(Z, p.wd) + p.bd)
            h2 = T.tanh(T.dot(h3, p.w3d) + p.b3d)
            h2 = h2.reshape((h2.shape[0], p.w2d.shape[1], downpool_sz, downpool_sz))
            h1 = rectify(deconv_and_depool(h2, p.w2d))
            pxz = T.nnet.sigmoid(deconv_and_depool(h1, p.w1d) + p.b1d )
            return pxz

        x = T.reshape(self.X, (-1, 1, 28, 28))
        x = binomial(x)

        mu_encoder, log_sigma_encoder = conv_enc(x, self.params)
        eps = srnd.normal(mu_encoder.shape, dtype=theano.config.floatX) 
        z = mu_encoder + T.exp(log_sigma_encoder)*eps
        pxz = deconv_dec(z, self.params)
        s_pxz = deconv_dec(self.Z, self.params)
        s_pxz = T.flatten(s_pxz, outdim=2)

        log_qpz = -0.5 * T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))
        log_pxz = T.nnet.binary_crossentropy(pxz,x).sum()
        cost = log_pxz + log_qpz

        out_log_qpz = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder), axis = 1)
        out_log_pxz = -T.nnet.binary_crossentropy(pxz,x).sum(axis = 1)

        a_pxz = T.zeros((self.n_t, s_pxz.shape[0], s_pxz.shape[1]))
        a_pxz = T.set_subtensor(a_pxz[0,:,:], s_pxz)

        self.compile(log_pxz, log_qpz, cost, a_pxz)

