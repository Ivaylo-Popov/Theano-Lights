import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

from toolbox import *
from modelbase import *




-- Train 1 day and gives 82 perplexity. -- Trains 1h and gives test 115 perplexity.
local params = {batch_size=20,  local params = {batch_size=20,
                seq_length=35,                  seq_length=20,
                layers=2,                       layers=2,
                decay=1.15,                     decay=2,
                rnn_size=1500,                  rnn_size=200,
                dropout=0.65,                   dropout=0,
                init_weight=0.04,               init_weight=0.1,
                lr=1,                           lr=1,
                vocab_size=10000,               vocab_size=10000,
                max_epoch=14,                   max_epoch=4,
                max_max_epoch=55,               max_max_epoch=13,
                max_grad_norm=10}               max_grad_norm=5}
               ]]--

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size, params.rnn_size)(x)}
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))

  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s)})
end

local words_per_step = params.seq_length * params.batch_size
local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
local perps
local perp = fp(state_train)
perps = torch.zeros(epoch_size):add(perp)
perps[step % epoch_size + 1] = perp
total_cases = total_cases + params.seq_length * params.batch_size
g_f3(torch.exp(perps:mean())) ..

class FFN(ModelSLBase):
    def __init__(self, data, hp):
        super(FFN, self).__init__(self.__class__.__name__, data, hp)
        
        self.n_h = 400

        self.params = Parameters()
        n_x = self.data['n_x']
        n_y = self.data['n_y']
        n_h = self.n_h
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                w_h = shared_normal((n_x, n_h), scale=scale)
                b_h = shared_normal((n_h,), scale=0)
                w_h2 = shared_normal((n_h, n_h), scale=scale)
                b_h2 = shared_normal((n_h,), scale=0)
                w_h3 = shared_normal((n_h, n_h), scale=scale)
                b_h3 = shared_normal((n_h,), scale=0)
                w_o = shared_normal((n_h, n_y), scale=scale)
        
        def model(X, params, p_drop_input, p_drop_hidden):
            X = dropout(X, p_drop_input)
        
            h = dropout(rectify(T.dot(X, params.w_h) + params.b_h ), p_drop_hidden)
            h2 = dropout(rectify(T.dot(h, params.w_h2) + params.b_h2), p_drop_hidden)
            h3 = dropout(rectify(T.dot(h2, params.w_h3) + params.b_h3), p_drop_hidden)

            py_x = softmax(T.dot(h3, params.w_o))
            return py_x
        
        noise_py_x = model(self.X, self.params, 0.2, 0.2)
        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, self.Y))
        
        py_x = model(self.X, self.params, 0., 0.)
        y_x = T.argmax(py_x, axis=1)

        self.compile(cost, y_x)

