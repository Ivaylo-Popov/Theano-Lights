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


class ModelSLBase(object):
    def __init__(self, id, data, hp):
        self.type = 'SL'
        self.id = id
        self.filename = 'savedmodels\model_'+id+'.pkl'
        self.hp = hp

        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        self.data = copy.copy(data)
        for key in ('tr_X', 'va_X', 'te_X', 'tr_Y', 'va_Y', 'te_Y'):
            if key in self.data:
                self.data[key] = shared(self.data[key], borrow=True)
        
        if hp['debug']:
            theano.config.optimizer = 'None'
            theano.config.compute_test_value = 'ignore'
            theano.config.exception_verbosity = 'high'
            

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)

    def permuteData(self, data=None):
        if self.hp.train_perm:
            perm_idx = np.random.permutation(self.data['P'])
            self.data['tr_X'].set_value(self.data['tr_X'].get_value(borrow=True)[perm_idx], borrow=True)
            self.data['tr_Y'].set_value(self.data['tr_Y'].get_value(borrow=True)[perm_idx], borrow=True)
            if not data is None:
                data['tr_X'] = data['tr_X'][perm_idx]
                data['tr_Y'] = data['tr_Y'][perm_idx]

    def compile(self, cost, y_x):
        batch_idx = T.iscalar()

        updates = self.hp.optimizer(cost, self.params.values(), lr=self.hp.learning_rate)

        self.train = theano.function(inputs=[batch_idx],
                                     givens={
                                         self.X:self.data['tr_X'][batch_idx * self.hp.batch_size : 
                                                                  (batch_idx+1) * self.hp.batch_size],
                                         self.Y:self.data['tr_Y'][batch_idx * self.hp.batch_size : 
                                                                  (batch_idx+1) * self.hp.batch_size]},
                                     outputs=[y_x], updates=updates)
        
        self.validate = theano.function(inputs=[], 
                                        givens={self.X:self.data['va_X']},
                                    outputs=[y_x])
        
        self.test = theano.function(inputs=[], 
                                    givens={self.X:self.data['te_X']},
                                    outputs=[y_x])

        
# --------------------------------------------------------------------------------------------------

class ModelULBase(object):
    def __init__(self, id, data, hp):
        self.max_gen_samples = 10000
        self.type = 'UL'
        self.id = id
        self.filename = 'savedmodels\model_'+id+'.pkl'
        self.hp = hp

        self.X = T.fmatrix('X')
        self.Z = T.fmatrix('Z')

        self.X.tag.test_value = np.random.randn(1000, 784).astype(dtype=theano.config.floatX)
        
        self.data = copy.copy(data)
        for key in ('tr_X', 'va_X', 'te_X'):
            if key in self.data:
                self.data[key] = shared(self.data[key], borrow=True)

        if hp['debug']:
            theano.config.optimizer = 'None'
            theano.config.compute_test_value = 'ignore'
            theano.config.exception_verbosity = 'high'

    def save(self):
        if not os.path.exists('savedmodels\\'):
            os.makedirs('savedmodels\\')
        self.params.save(self.filename)

    def permuteData(self):
        if self.hp.train_perm:
            self.data['tr_X'].set_value(self.data['tr_X'].get_value(borrow=True)[np.random.permutation(self.data['P'])], borrow=True)

    def compile(self, log_pxz, log_qpz, cost, a_pxz):
        batch_idx = T.iscalar()

        updates = self.hp.optimizer(cost, self.params.values(), lr=self.hp.learning_rate)

        self.train = theano.function(inputs=[batch_idx], 
                                     givens={self.X:self.data['tr_X'][batch_idx * self.hp.batch_size : 
                                                                      (batch_idx+1) * self.hp.batch_size]},
                                     outputs=[log_pxz, log_qpz], updates=updates)
        
        self.validate = theano.function(inputs=[], 
                                        givens={self.X:self.data['va_X']},
                                        outputs=[log_pxz, log_qpz])
        
        self.test = theano.function(inputs=[], 
                                    givens={self.X:self.data['te_X']},
                                    outputs=[log_pxz, log_qpz])

        self.testBatch = theano.function(inputs=[batch_idx], 
                                         givens={self.X:self.data['te_X'][batch_idx * self.hp.test_batch_size : 
                                                                          (batch_idx+1) * self.hp.test_batch_size]}, 
                                         outputs=[log_pxz, log_qpz])
        
        n_samples = T.iscalar()

        if self.hp.resample_z:
            self.data['ge_Z'] = srnd.normal((self.max_gen_samples, self.n_z), dtype=theano.config.floatX)
        else:
            self.data['ge_Z'] = shared(np.random.randn(self.max_gen_samples, self.n_z))

        self.decode = theano.function(inputs=[n_samples], 
                                      givens={self.Z:self.data['ge_Z'][:n_samples]}, 
                                      outputs=a_pxz)

