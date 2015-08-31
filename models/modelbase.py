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
from operator import add

from toolbox import *


class ModelBase(object):
    def train_epoch(self, it_lr):
        tr_outputs = None

        for i in xrange(0, self.data['len_tr_X'] / self.hp.batch_size):
            outputs = self.train(i, it_lr)
            outputs = map(lambda x: x / float(self.data['len_tr_X']), outputs)
            if i==0:
                tr_outputs = outputs
            else:
                tr_outputs = map(add, tr_outputs, outputs)
        return tr_outputs

    def test_epoch(self):
        te_outputs = None

        for i in xrange(0, self.data['len_te_X'] / self.hp.test_batch_size):
            outputs = self.test(i)
            outputs = map(lambda x: x / float(self.data['len_te_X']), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def validation_epoch(self):
        te_outputs = None

        for i in xrange(0, self.data['len_va_X'] / self.hp.test_batch_size):
            outputs = self.validate(i)
            outputs = map(lambda x: x / float(self.data['len_va_X']), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def train_walkstep(self, walkstep, ws_iterations, it_lr):
        tr_outputs = None

        for it in range(ws_iterations):
            for i in xrange(0, self.hp.walkstep_size):
                batch_idx = walkstep * self.hp.walkstep_size + i
                outputs = self.train(batch_idx, it_lr)
                outputs = map(lambda x: x / float(self.hp.walkstep_size * self.hp.batch_size), outputs)
                if it==0:
                    if i==0:
                        tr_outputs = outputs
                    else:
                        tr_outputs = map(add, tr_outputs, outputs)
        return tr_outputs

    def load(self):
        if os.path.isfile(self.filename):
                self.params.load(self.filename)

# --------------------------------------------------------------------------------------------------

class ModelSLBase(ModelBase):
    def __init__(self, id, data, hp):
        self.type = 'SL'
        self.id = id
        self.filename = 'savedmodels\model_'+id+'.pkl'
        self.hp = hp

        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        self.X.tag.test_value = np.random.randn(hp.batch_size, data['n_x']).astype(dtype=theano.config.floatX)

        self.data = copy.copy(data)
        for key in ('tr_X', 'va_X', 'te_X', 'tr_Y', 'va_Y', 'te_Y'):
            if key in self.data:
                self.data['len_'+key] = len(self.data[key])
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

    def compile(self, cost, error_map_pyx, add_updates=[], debug_info=[]):
        batch_idx = T.iscalar()
        learning_rate = T.fscalar()

        updates, norm_grad = self.hp.optimizer(cost, self.params.values(), lr=learning_rate)

        updates += add_updates

        self.outidx = {'cost':0, 'error_map_pyx':1, 'norm_grad':2}
        outputs = [cost, error_map_pyx]

        self.train = theano.function(inputs=[batch_idx, learning_rate], updates=updates,
                                     givens={
                                         self.X:self.data['tr_X'][batch_idx * self.hp.batch_size : 
                                                                  (batch_idx+1) * self.hp.batch_size],
                                         self.Y:self.data['tr_Y'][batch_idx * self.hp.batch_size : 
                                                                  (batch_idx+1) * self.hp.batch_size]},
                                     outputs=outputs + [norm_grad])
                                     #,mode=theano.compile.nanguardmode.NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        #T.printing.debugprint(self.train)
        #T.printing.pydotprint(self.train, outfile="logreg_pydotprint_train.png", var_with_name_simple=True)
        
        self.validate = theano.function(inputs=[batch_idx], 
                                        givens={
                                         self.X:self.data['va_X'][batch_idx * self.hp.test_batch_size : 
                                                                  (batch_idx+1) * self.hp.test_batch_size],
                                         self.Y:self.data['va_Y'][batch_idx * self.hp.test_batch_size : 
                                                                  (batch_idx+1) * self.hp.test_batch_size]},
                                    outputs=outputs)
        
        self.test = theano.function(inputs=[batch_idx], 
                                    givens={
                                         self.X:self.data['te_X'][batch_idx * self.hp.test_batch_size : 
                                                                  (batch_idx+1) * self.hp.test_batch_size],
                                         self.Y:self.data['te_Y'][batch_idx * self.hp.test_batch_size : 
                                                                  (batch_idx+1) * self.hp.test_batch_size]},
                                    outputs=outputs)

# --------------------------------------------------------------------------------------------------

class ModelULBase(ModelBase):
    def __init__(self, id, data, hp):
        self.max_gen_samples = 10000
        self.type = 'UL'
        self.id = id
        self.filename = 'savedmodels\model_'+id+'.pkl'
        self.hp = hp

        self.resample_z = False

        self.X = T.fmatrix('X')
        self.Z = T.fmatrix('Z')

        self.X.tag.test_value = np.random.randn(hp.batch_size, data['n_x']).astype(dtype=theano.config.floatX)
        
        self.data = copy.copy(data)
        for key in ('tr_X', 'va_X', 'te_X'):
            if key in self.data:
                self.data['len_'+key] = len(self.data[key])
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
            if not data is None:
                data['tr_X'] = data['tr_X'][perm_idx]

    def compile(self, log_pxz, log_qpz, cost, a_pxz):
        batch_idx = T.iscalar()
        learning_rate = T.fscalar()

        updates, norm_grad = self.hp.optimizer(cost, self.params.values(), lr=learning_rate)

        self.outidx = {'cost':0, 'cost_p':1, 'cost_q':2, 'norm_grad':3}
        outputs = [cost, log_pxz, log_qpz]

        self.train = theano.function(inputs=[batch_idx, learning_rate], 
                                     givens={self.X:self.data['tr_X'][batch_idx * self.hp.batch_size : 
                                                                      (batch_idx+1) * self.hp.batch_size]},
                                     outputs=outputs + [norm_grad], updates=updates)
        
        self.validate = theano.function(inputs=[batch_idx], 
                                        givens={self.X:self.data['tr_X'][batch_idx * self.hp.test_batch_size : 
                                                                      (batch_idx+1) * self.hp.test_batch_size]},
                                        outputs=outputs)
        
        self.test = theano.function(inputs=[batch_idx], 
                                    givens={self.X:self.data['te_X'][batch_idx * self.hp.test_batch_size : 
                                                                      (batch_idx+1) * self.hp.test_batch_size]},
                                    outputs=outputs)
        
        n_samples = T.iscalar()

        if self.resample_z:
            self.data['ge_Z'] = srnd.normal((self.max_gen_samples, self.n_z), dtype=theano.config.floatX)
        else:
            self.data['ge_Z'] = shared(np.random.randn(self.max_gen_samples, self.n_z))

        self.decode = theano.function(inputs=[n_samples], 
                                      givens={self.Z:self.data['ge_Z'][:n_samples]}, 
                                      outputs=a_pxz)

# --------------------------------------------------------------------------------------------------

class ModelLMBase(ModelBase):
    def __init__(self, id, data, hp):
        self.type = 'LM'
        self.id = id
        self.filename = 'savedmodels\model_'+id+'.pkl'
        self.hp = hp

        self.X = T.imatrix()
        self.Y = T.ivector()
        self.seed_idx = T.iscalar()

        self.X.tag.test_value = np.random.randn(hp.seq_size, hp.batch_size).astype(dtype=np.int32)

        self.data = copy.copy(data)
        for key in ('tr_X', 'va_X', 'te_X', 'tr_Y', 'va_Y', 'te_Y'):
            if key in self.data:
                self.data['len_'+key] = len(self.data[key])
                self.data[key] = shared(self.data[key], borrow=True, dtype=np.int32)
        
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

    def reset_hiddenstates(self):
        for hs in self.hiddenstates.values():
            hs = hs * 0.0

    def train_epoch(self, it_lr, offset=0):
        tr_outputs = None
        seq_per_epoch = self.hp.batch_size * (self.hp.seq_size - self.hp.warmup_size) * (self.data['len_tr_X'] - offset) / self.hp.seq_size

        self.reset_hiddenstates()
        
        for i in xrange(0, (self.data['len_tr_X'] - offset) / self.hp.seq_size):
            outputs = self.train(i, it_lr, offset)
            outputs = map(lambda x: x / float(seq_per_epoch), outputs)
            if i==0:
                tr_outputs = outputs
            else:
                tr_outputs = map(add, tr_outputs, outputs)
        return tr_outputs

    def validation_epoch(self):
        te_outputs = None
        seq_per_epoch = self.hp.batch_size * (self.hp.seq_size - self.hp.warmup_size) * self.data['len_va_X'] / self.hp.seq_size

        self.reset_hiddenstates()
        
        for i in xrange(0, self.data['len_va_X'] / self.hp.seq_size):
            outputs = self.validate(i)
            outputs = map(lambda x: x / float(seq_per_epoch), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def test_epoch(self):
        te_outputs = None
        seq_per_epoch = self.hp.batch_size * (self.hp.seq_size - self.hp.warmup_size) * self.data['len_te_X'] / self.hp.seq_size

        self.reset_hiddenstates()

        for i in xrange(0, self.data['len_te_X'] / self.hp.seq_size):
            outputs = self.test(i)
            outputs = map(lambda x: x / float(seq_per_epoch), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def dyn_validation_epoch(self, it_lr):
        te_outputs = None
        seq_per_epoch = self.hp.batch_size * (self.hp.seq_size - self.hp.warmup_size) * self.data['len_va_X'] / self.hp.seq_size

        self.reset_hiddenstates()
        
        for i in xrange(0, self.data['len_va_X'] / self.hp.seq_size):
            outputs = self.dyn_validate(i, it_lr)
            outputs = map(lambda x: x / float(seq_per_epoch), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def dyn_test_epoch(self, it_lr):
        te_outputs = None
        seq_per_epoch = self.hp.batch_size * (self.hp.seq_size - self.hp.warmup_size) * self.data['len_te_X'] / self.hp.seq_size

        self.reset_hiddenstates()

        for i in xrange(0, self.data['len_te_X'] / self.hp.seq_size):
            outputs = self.dyn_test(i, it_lr)
            outputs = map(lambda x: x / float(seq_per_epoch), outputs)
            if te_outputs is None:
                te_outputs = outputs
            else:
                te_outputs = map(add, te_outputs, outputs)
        return te_outputs

    def compile(self, cost, te_cost, h_updates, te_h_updates, add_updates=[]):
        seq_idx = T.iscalar()
        learning_rate = T.fscalar()
        offset = T.iscalar()

        updates, norm_grad = self.hp.optimizer(cost, self.params.values(), lr=learning_rate)
        
        updates += add_updates

        self.outidx = {'cost':0, 'norm_grad':1}

        self.train = theano.function(inputs=[seq_idx, learning_rate, offset], updates=updates + h_updates,
                                     givens={
                                         self.X:self.data['tr_X'][offset + seq_idx * self.hp.seq_size : 
                                                                  offset + (seq_idx+1) * self.hp.seq_size]},
                                     outputs=[cost, norm_grad])
        
        self.validate = theano.function(inputs=[seq_idx], updates=te_h_updates,
                                        givens={
                                            self.X:self.data['va_X'][seq_idx * self.hp.seq_size : 
                                                                    (seq_idx+1) * self.hp.seq_size]},
                                    outputs=[te_cost])
        
        self.test = theano.function(inputs=[seq_idx], updates=te_h_updates,
                                    givens={
                                            self.X:self.data['te_X'][seq_idx * self.hp.seq_size : 
                                                                    (seq_idx+1) * self.hp.seq_size]},
                                    outputs=[te_cost])

        if self.hp.dynamic_eval:
            self.dyn_validate = theano.function(inputs=[seq_idx, learning_rate], updates=updates + te_h_updates,
                                        givens={
                                         self.X:self.data['va_X'][seq_idx * self.hp.seq_size : 
                                                                  (seq_idx+1) * self.hp.seq_size]},
                                        outputs=[te_cost])
        
            self.dyn_test = theano.function(inputs=[seq_idx, learning_rate], updates=updates + te_h_updates,
                                        givens={
                                             self.X:self.data['te_X'][seq_idx * self.hp.seq_size : 
                                                                      (seq_idx+1) * self.hp.seq_size]},
                                        outputs=[te_cost])

# --------------------------------------------------------------------------------------------------
