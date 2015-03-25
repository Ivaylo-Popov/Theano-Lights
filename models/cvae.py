import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *

def padBoth(x, n_p):
    sx = int(np.sqrt(x.shape[-1]))
    x = np.reshape(x, (-1, 1, sx, sx))
    x = np.concatenate([
        np.zeros((x.shape[0], 1, n_p, sx), dtype=theano.config.floatX), 
        x, 
        np.zeros((x.shape[0], 1, n_p, sx), dtype=theano.config.floatX)], axis=2)
    x = np.concatenate([
        np.zeros((x.shape[0], 1, sx + 4, n_p), dtype=theano.config.floatX), 
        x, 
        np.zeros((x.shape[0], 1, sx + 4, n_p), dtype=theano.config.floatX)], axis=3)
    return x

class Cvae(ModelULBase):
    def __init__(self, data, hp):
        
        data['shape_x'] = (data['shape_x'][0]+4, data['shape_x'][1]+4)
        data['tr_X'] = padBoth(data['tr_X'], 2)
        data['va_X'] = padBoth(data['va_X'], 2)
        data['te_X'] = padBoth(data['te_X'], 2)

        super(Cvae, self).__init__(self.__class__.__name__, data, hp)

        self.n_h = 512
        self.n_z = 20
        self.n_t = 1

        self.gaussian = False
            
        self.params = Parameters()
        
        dim_x = 32
        n_h = self.n_h
        n_z = self.n_z
        n_t = self.n_t
        scale = hp.init_scale
        downpool_sz = dim_x // 8   # assume square 2D, 2 layers mean downsampling by 2 ** 2 -> 4

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            l1_e = (16, 1, 3, 3)
            l2_e = (32, l1_e[0], 3, 3)
            l3_e = (64, l2_e[0], 3, 3)
            l4_e = (l3_e[0] * downpool_sz * downpool_sz, n_h)
            lz_e = (n_h, n_z)
            l1_d = (l1_e[1], l1_e[0], l1_e[2], l1_e[3])
            l2_d = (l2_e[1], l2_e[0], l2_e[2], l2_e[3])
            l3_d = (l3_e[1], l3_e[0], l3_e[2], l3_e[3])
            l4_d = (l4_e[1], l4_e[0])
            lz_d = (lz_e[1], lz_e[0])

            with self.params:
                w1e = shared_normal(l1_e, scale=scale)
                w2e = shared_normal(l2_e, scale=scale)
                w3e = shared_normal(l3_e, scale=scale)
                w4e = shared_normal(l4_e, scale=scale)
                b4e = shared_zeros(l4_e[1])
                wmu = shared_normal(lz_e, scale=scale)
                bmu = shared_zeros(n_z)
                wsigma = shared_normal(lz_e, scale=scale)
                bsigma = shared_zeros(n_z)

                wd = shared_normal(lz_d, scale=scale)
                bd = shared_zeros((lz_d[1]))
                w4d = shared_normal(l4_d, scale=scale)
                b4d = shared_zeros((l4_d[1]))
                w3d = shared_normal(l3_d, scale=scale)
                w2d = shared_normal(l2_d, scale=scale)
                w1d = shared_normal(l1_d, scale=scale)
                b1d = shared_normal((1, dim_x, dim_x))
        
        def conv_enc(X, p):
            h1 = rectify(max_pool_2d(conv(X, p.w1e), (2, 2)))
            h2 = rectify(max_pool_2d(conv(h1, p.w2e), (2, 2)))
            h3 = rectify(max_pool_2d(conv(h2, p.w3e), (2, 2)))
            h3 = T.flatten(h3, outdim=2)
            h4 = T.tanh(T.dot(h3, p.w4e) + p.b4e)
            mu = T.dot(h4, p.wmu) + p.bmu
            log_sigma = 0.5 * (T.dot(h4, p.wsigma) + p.bsigma)
            return mu, log_sigma

        def deconv_dec(Z, p):
            h3 = T.tanh(T.dot(Z, p.wd) + p.bd)
            h2 = T.tanh(T.dot(h3, p.w4d) + p.b4d)
            h2 = h2.reshape((h2.shape[0], p.w3d.shape[1], downpool_sz, downpool_sz))
            h1 = rectify(deconv_and_depool(h2, p.w3d))
            h0 = rectify(deconv_and_depool(h1, p.w2d))
            pxz = T.nnet.sigmoid(deconv_and_depool(h0, p.w1d) + p.b1d )
            return pxz

        self.X = T.ftensor4('X')
        x = binomial(self.X)

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

