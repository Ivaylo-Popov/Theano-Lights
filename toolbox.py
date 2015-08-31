import os
try:
    import cPickle as pickle
except:
    import pickle
from PIL import Image
from subprocess import Popen
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import gzip
from random import shuffle
from collections import OrderedDict
import inspect
import h5py
import sys
from skimage.measure import block_reduce

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams

sys.setrecursionlimit(100000)

#--------------------------------------------------------------------------------------------------

srnd = MRG_RandomStreams()


def shared(X, name=None, dtype=theano.config.floatX, borrow=False, broadcastable=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name, borrow=borrow, broadcastable=broadcastable)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def rectify(X):
    return T.maximum(X, 0.)

def leakyrectify(X, alpha=0.05):
    return T.maximum(X, alpha*X)

def cliplin(X):
    return T.minimum(T.maximum(X, -2.), 2.)

def binomial(X):
    return srnd.binomial(X.shape, p=X, dtype=theano.config.floatX)

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srnd.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def gdropout(X, p=0.):
    if p > 0:
        X *= srnd.normal(X.shape, avg=1, std=p, dtype=theano.config.floatX)
    return X

def gaussian(shape, std=0.):
    if std > 0:
        return srnd.normal(shape, std = std, dtype=theano.config.floatX)
    else:
        return T.zeros_like(shape, dtype=theano.config.floatX)

def shared_zeros(shape, dtype=theano.config.floatX, name=None, broadcastable=None):
    return shared(np.zeros(shape), dtype=dtype, name=name, broadcastable=broadcastable)

def shared_uniform(shape, scale=0.05):
    return shared(np.random.uniform(low=-scale, high=scale, size=shape))

def shared_uniform(shape, range=[-0.05,0.05]):
    return shared(np.random.uniform(low=range[0], high=range[1], size=shape))

def shared_normal(shape, sv_adjusted=True, scale=1.0, name=None):
    if (sv_adjusted):
        if len(shape) == 1:
            scale_factor =  scale / np.sqrt(shape[0])
        elif len(shape) == 2:
            scale_factor =  scale / (np.sqrt(shape[0]) + np.sqrt(shape[1]))
        else:
            scale_factor =  scale / (np.sqrt(shape[1]) + np.sqrt(shape[2]))
    else:
        scale_factor = scale    
    if len(shape) == 2:
        return shared(np.random.randn(*shape) * scale_factor, name=name, broadcastable=(shape[0]==1,shape[1]==1))
    else:
        return shared(np.random.randn(*shape) * scale_factor, name=name)

def batched_dot(A, B):
    return (A[:,:,:,None]*B[:,None,:,:]).sum(axis=-2)

def concatenate(tensor_list, axis=0):
    if axis < 0:
        axis += tensor_list[0].ndim

    concat_size = sum(tensor.shape[axis] for tensor in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tensor in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tensor.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tensor)
        offset += tensor.shape[axis]

    return out

def batch_col(input_size, c):
    return T.zeros((input_size, c.shape[0])) + c

def normalize(v):
    v = v / (1e-6 + T.max(T.abs_(v), axis=1, keepdims=True))
    v_2 = T.sum(v**2, axis=1, keepdims=True)
    return v / T.sqrt(1e-6 + v_2)

#--------------------------------------------------------------------------------------------------

def norm_gs(params, grads):
    norm_gs = 0.
    for g in grads:
        norm_gs += (g**2).sum()
    
    return norm_gs

def sgd(cost, params, lr=1.0, alpha=0.1):
    """SGD with Momentum (and Langevin Dynamics)"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0.)
        v_new = v * (1.0 - alpha) - alpha * lr * g
        updates.append((v, v_new))
        updates.append((p, p + v_new ))  #+ T.sqrt(lr) / 600.0 * srnd.normal(v.shape, dtype=theano.config.floatX)
    
    return updates, norm_gs(params, grads)

def sgdgc(cost, params, lr=1.0, max_magnitude=5.0, infDecay=0.1):
    """SGD with gradient clipping"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    norm = norm_gs(params, grads)
    sqrtnorm = T.sqrt(norm)
    #not_finite = T.or_(T.isnan(sqrtnorm), T.isinf(sqrtnorm))
    adj_norm_gs = T.switch(T.ge(sqrtnorm, max_magnitude), max_magnitude / sqrtnorm, 1.)

    for p, g in zip(params, grads):
        #g = T.switch(not_finite, infDecay * p, g * adj_norm_gs)
        updates.append((p, p - lr * g * adj_norm_gs))  
    
    return updates, norm

def sgdmgc(cost, params, lr=1.0, alpha=0.1, max_magnitude=5.0, infDecay=0.1):
    """SGD with momentum and gradient clipping"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    norm = norm_gs(params, grads)
    sqrtnorm = T.sqrt(norm)
    not_finite = T.or_(T.isnan(sqrtnorm), T.isinf(sqrtnorm))
    adj_norm_gs = T.switch(T.ge(sqrtnorm, max_magnitude), max_magnitude / sqrtnorm, 1.)

    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0.)
        g = T.switch(not_finite, infDecay * p, g * adj_norm_gs)
        v_new = v * (1.0 - alpha) - alpha * lr * g
        updates.append((v, v_new))
        updates.append((p, p + v_new ))
    
    return updates, norm

def rmsprop(cost, params, lr=0.001, alpha=0.2, beta=0.2, epsilon=1e-6, decay_rate=0.0, data_part=0.0): 
    """RMSprop and AdaGrad"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0.)
        acc_new = acc - beta * acc + alpha * g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - g / T.sqrt(acc_new + epsilon) * lr - p * decay_rate * data_part))
    return updates, norm_gs(params, grads)

def esgd(cost, params, lr=0.02, e=1e-2): 
    """Equilibrated SGD"""
    updates = []
    grads = T.grad(cost=cost, wrt=params)
    i = shared(floatX(0.))
    i_t = i + 1.
    updates.append((i, i_t))
    ss = 0
    for p, g in zip(params, grads):
        v = srnd.normal(g.shape, dtype=theano.config.floatX) 
        ss += T.sum(g * v)
    vH = T.grad(ss, params)
    for p, g, gg in zip(params, grads, vH):
        acc = shared(p.get_value() * 0.)
        acc_new = acc + gg ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - g / (T.sqrt(acc_new/i_t) + e) * lr))
    return updates, norm_gs(params, grads)

def adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = shared(p.get_value() * 0.)
        v = shared(p.get_value() * 0.)
        #g = g + srnd.normal(g.shape, avg = 0.0, std = 0.01, dtype=theano.config.floatX)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        #m_t += srnd.normal(m_t.shape, std = 0.01, dtype=theano.config.floatX)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates, norm_gs(params, grads)

def adamgc_(cost, params, lr=0.0002, b1=0.1, b2=0.01, e=1e-8, max_magnitude=5.0, infDecay=0.1):
    updates = []
    grads = T.grad(cost, params)
    
    norm = norm_gs(params, grads)
    sqrtnorm = T.sqrt(norm)
    not_finite = T.or_(T.isnan(sqrtnorm), T.isinf(sqrtnorm))
    adj_norm_gs = T.switch(T.ge(sqrtnorm, max_magnitude), max_magnitude / sqrtnorm, 1.)

    i = shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        g = T.switch(not_finite, infDecay * p, g * adj_norm_gs)
        m = shared(p.get_value() * 0.)
        v = shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m) 
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)

        #e_t = shared(p.get_value() * 0.)
        #de_t = (srnd.normal(p.shape, std = 0.05, dtype=theano.config.floatX)*p_t - e_t)*0.05  #*p_t
        #p_t = p_t + de_t
        #updates.append((e_t, e_t + de_t))

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates, norm

def adamgc(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, max_magnitude=5.0, infDecay=0.1):
    updates = []
    grads = T.grad(cost, params)
    
    norm = norm_gs(params, grads)
    sqrtnorm = T.sqrt(norm)
    not_finite = T.or_(T.isnan(sqrtnorm), T.isinf(sqrtnorm))
    adj_norm_gs = T.switch(T.ge(sqrtnorm, max_magnitude), max_magnitude / sqrtnorm, 1.)

    i = shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        g = T.switch(not_finite, infDecay * p, g * adj_norm_gs)
        m = shared(p.get_value() * 0.)
        v = shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m) 
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates, norm

#--------------------------------------------------------------------------------------------------

def theano_one_hot(idxs, n):
    z = T.zeros((idxs.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idxs.shape[0]), idxs], 1)
    return one_hot

def one_hot(x, n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def shuffledata(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [np.matrix([d[idx] for idx in idxs]) for d in data]

def concatdata(trX,vaX,teX=None):
    if teX==None:
        fullX = np.zeros((len(trX)+len(vaX),trX.shape[1]), dtype=float)
        fullX[:len(trX)] = trX
        fullX[len(trX):len(trX)+len(vaX)] = vaX
        return fullX
    else:
        fullX = np.zeros((len(trX)+len(vaX)+len(teX),trX.shape[1]), dtype=float)
        fullX[:len(trX)] = trX
        fullX[len(trX):len(trX)+len(vaX)] = vaX
        fullX[len(trX)+len(vaX):len(trX)+len(vaX)+len(teX)] = teX
        return fullX

def downsample(data):
    data['_tr_X'] = np.zeros((len(data['tr_X']), 14*14), dtype='float32')
    data['_va_X'] = np.zeros((len(data['va_X']), 14*14), dtype='float32')
    data['_te_X'] = np.zeros((len(data['te_X']), 14*14), dtype='float32')

    for i in xrange(0, len(data['tr_X'])):
        data['_tr_X'][i] = block_reduce(data['tr_X'][i].reshape(data['shape_x']), block_size=(2,2), func=np.mean).flatten() 

    for i in xrange(0, len(data['va_X'])):
        data['_va_X'][i] = block_reduce(data['va_X'][i].reshape(data['shape_x']), block_size=(2,2), func=np.mean).flatten() 

    for i in xrange(0, len(data['te_X'])):
        data['_te_X'][i] = block_reduce(data['te_X'][i].reshape(data['shape_x']), block_size=(2,2), func=np.mean).flatten() 

    data['tr_X'] = data['_tr_X']
    data['va_X'] = data['_va_X']
    data['te_X'] = data['_te_X']

    data['shape_x'] = (14,14)
    data['n_x'] = 14*14
    return data

def freyfaces(path='', distort=False,shuffle=False,ntrain=60000,ntest=10000,onehot=True):
    f = open(os.path.join(path,'freyfaces.pkl'),'rb')
    data = pickle.load(f)
    f.close()

    lenX = len(data) * 0.9

    trX = data[:lenX,:]
    trY = data[:lenX,:1]
    teX = data[lenX:,:]
    teY = data[lenX:,:1]

    data = {}
    data['tr_P'] = len(trX)
    data['n_x'] = int(trX.shape[1])
    data['n_y'] = int(trY.shape[1])
    data['shape_x'] = (28,20)
    
    data['tr_X'] = trX.astype('float32'); 
    data['te_X'] = teX.astype('float32'); 
    data['tr_Y'] = trY.astype('float32'); 
    data['te_Y'] = teY.astype('float32'); 
    
    return data

def mnistPkl(filename):
    train_set_x = numpy.concatenate((train_set_x, valid_set_x), axis=0)
    train_set_y = numpy.concatenate((train_set_y, valid_set_y), axis=0)

def mnistBinarized(path=''):
    train_x = h5py.File(path+"binarized_mnist-train.h5")['data'][:]
    valid_x = h5py.File(path+"binarized_mnist-valid.h5")['data'][:]
    test_x = h5py.File(path+"binarized_mnist-test.h5")['data'][:]

    data = {}
    data['P'] = len(train_x)
    data['n_x'] = int(train_x.shape[1])
    data['n_y'] = 0
    data['shape_x'] = (28,28)
    
    data['tr_X'] = train_x.astype('float32'); 
    data['va_X'] = valid_x.astype('float32'); 
    data['te_X'] = test_x.astype('float32'); 
    
    return data

def mnist2(path=''):
    filepath = os.path.join(path,'mnist.pkl.gz')
    f = gzip.open(filepath, 'rb')
    (trX,trY),(vaX,vaY),(teX,teY) = pickle.load(f)
    f.close()

    trX = np.concatenate((trX, vaX), axis=0)
    trY = np.concatenate((trY, vaY), axis=0)
    vaX = vaX[1:0]
    vaY = vaY[1:0]
    
    trY = one_hot(trY, 10)
    vaY = one_hot(vaY, 10)
    teY = one_hot(teY, 10)

    data = {}
    data['P'] = len(trX)
    data['n_x'] = int(trX.shape[1])
    data['n_y'] = int(trY.shape[1])
    data['shape_x'] = (28,28)

    data['tr_X'] = trX.astype('float32'); 
    data['va_X'] = vaX.astype('float32'); 
    data['te_X'] = teX.astype('float32'); 
    data['tr_Y'] = trY.astype('float32'); 
    data['va_Y'] = vaY.astype('float32'); 
    data['te_Y'] = teY.astype('float32');

    return data

def mnist(path='', distort=0,shuffle=False,nvalidation=10000):
	if distort!=0:
		ninst = 60000*(1 + distort)
		ntrain = ninst
		fd = open(os.path.join(path,'train-images-idx3-ubyte_distorted'))
	else:
		ninst = 60000
		fd = open(os.path.join(path,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:ninst*784+16].reshape((ninst,28*28)).astype(float)

	if distort!=0:
		fd = open(os.path.join(path,'train-labels-idx1-ubyte_distorted'))
	else:
		fd = open(os.path.join(path,'train-labels-idx1-ubyte'))
	
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:ninst+8].reshape((ninst))

	fd = open(os.path.join(path,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(path,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX /= 255.
	teX /= 255.
    
	ntrain = ninst-nvalidation*(1 + distort)
	vaX = trX[ntrain:ninst]
	vaY = trY[ntrain:ninst]
	trX = trX[0:ntrain]
	trY = trY[0:ntrain]

	if shuffle:
		idx = np.random.permutation(ntrain)
		trX_n = trX
		trY_n = trY
		for i in range(ntrain):
			trX[i] = trX_n[idx[i]]
			trY[i] = trY_n[idx[i]]
		trX_n = None
		trY_n = None

	trY = one_hot(trY, 10)
	vaY = one_hot(vaY, 10)
	teY = one_hot(teY, 10)

	data = {}
	data['P'] = len(trX)
	data['n_x'] = int(trX.shape[1])
	data['n_y'] = int(trY.shape[1])
	data['shape_x'] = (28,28)

	data['tr_X'] = trX.astype('float32'); 
	data['va_X'] = vaX.astype('float32'); 
	data['te_X'] = teX.astype('float32'); 
	data['tr_Y'] = trY.astype('float32'); 
	data['va_Y'] = vaY.astype('float32'); 
	data['te_Y'] = teY.astype('float32');

	return data

def tokentext(name, path='', batch_size=100, data_mode='words', n_train=0):
    # data_mode in ('words', 'chars')

    def slice_batches(data_x, seq_size):
        size = (len(data_x) / batch_size) * batch_size
        batch_data = data_x[:size].reshape(batch_size, -1).transpose()
        return batch_data

    npz_data = np.load(path + name + ".npz")
    
    data = {}
    
    data['tr_X'] = slice_batches(npz_data['train_' + data_mode][30:], batch_size)
    if n_train != 0:
        data['tr_X'] = data['tr_X'][:n_train/batch_size]

    data['va_X'] = slice_batches(npz_data['valid_' + data_mode], batch_size)
    data['te_X'] = slice_batches(npz_data['test_' + data_mode], batch_size)

    data['P'] = len(data['tr_X'])
    data['n_x'] = int(1)
    data['shape_x'] = (1, batch_size)
    data['n_tokens'] = int(npz_data['n_words'])
    
    #npz_data = np.load(path + name + "_dict.npz")
    #data['vocabulary'] = npz_data['unique_' + data_mode]

    return data

def text_fromtokens(token1hot, vocabulary):
    text = ''
    for i in xrange(token1hot.shape[0]):
        for j in xrange(token1hot.shape[1]):
            if token1hot[i,j] > 0 and not vocabulary[j] is None:
                text += vocabulary[j] + ' '
                break
    return text

#--------------------------------------------------------------------------------------------------

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_=False, output_pixvals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixvals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixvals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct dtype
                dt = out_array.dtype
                if output_pixvals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_, output_pixvals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixvals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_:
                        this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the output array
                    c = 1
                    if output_pixvals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

def visualize_tokens(it, images, shape, p=0):
    image_data = tile_raster_images(images, img_shape=[shape[0], shape[1]], tile_shape=[len(images)/shape[0], shape[0]], tile_spacing=(0,0))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('samples_'+str(it)+'.png')

def visualize(it, images, shape, p=0):
    image_data = tile_raster_images(images, img_shape=[shape[0], shape[1]], tile_shape=[np.sqrt(len(images)).astype('int32'),len(images)/np.sqrt(len(images)).astype('int32')], tile_spacing=(2,2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('samples_'+str(it)+'.png')
    #if (p != 0):
    #    p.terminate()
    #return Popen(['mspaint.exe', 'samples_'+str(it)+'.png']) 

#--------------------------------------------------------------------------------------------------

class Parameters():
	def __init__(self):
		self.__dict__['params'] = {}
	
	def __setattr__(self,name,array):
		params = self.__dict__['params']
		if name not in params:
			params[name] = array
	
	def __setitem__(self,name,array):
		self.__setattr__(name,array)
	
	def __getitem__(self,name):
		return self.__getattr__(name)
	
	def __getattr__(self,name):
		params = self.__dict__['params']
		return self.params[name]
	
	def remove(self,name):
		del self.__dict__['params'][name]

	def get(self):
		return self.__dict__['params']

	def values(self):
		params = self.__dict__['params']
		return params.values()

	def save(self,filename):
		params = self.__dict__['params']
		pickle.dump({p:params[p] for p in params},open(filename,'wb'),2)

	def load(self,filename):
		params = self.__dict__['params']
		loaded = pickle.load(open(filename,'rb'))
		for k in loaded:
			params[k] = loaded[k]

	def setvalues(self, values):
		params = self.__dict__['params']
		for p, v in zip(params, values):
			params[p] = v

	def __enter__(self):
		_,_,_,env_locals = inspect.getargvalues(inspect.currentframe().f_back)
		self.__dict__['_env_locals'] = env_locals.keys()

	def __exit__(self,type,value,traceback):
		_,_,_,env_locals = inspect.getargvalues(inspect.currentframe().f_back)
		prev_env_locals = self.__dict__['_env_locals']
		del self.__dict__['_env_locals']
		for k in env_locals.keys():
			if k not in prev_env_locals:
				self.__setattr__(k,env_locals[k])
				env_locals[k] = self.__getattr__(k)
		return True

#--------------------------------------------------------------------------------------------------
def conv(X, w, b=None):
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z

def deconv(X, w, b=None):
    # z = dnn_conv(X, w, direction_hint="*not* 'forward!", border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z

def depool(X, factor=2):
    """
    Luke perforated upsample: http://www.brml.org/uploads/tx_sibibtex/281.pdf
    """
    output_shape = [
        X.shape[1],
        X.shape[2]*factor,
        X.shape[3]*factor
    ]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows*factor + (rows/stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0], output_shape[1], output_shape[2]))

    return upsamp

def deconv_and_depool(X, w, b=None):
    return deconv(depool(X), w, b)

#--------------------------------------------------------------------------------------------------

class AttentionDraw(object):
    def __init__(self, img_height, img_width, N):
        self.n_att_params = 5
        self.img_height = img_height
        self.img_width = img_width
        self.N = N
        self.a = T.arange(self.img_width)
        self.b = T.arange(self.img_height)
        self.rngN = T.arange(self.N) - self.N/2 - 0.5
        self.numtol = 1e-4
        self.delta_factor = (max(self.img_width, self.img_height)-1) / (self.N-1)
        self.center_x_factor = (self.img_width+1.) /2.
        self.center_y_factor = (self.img_height+1.) /2.

    def filterbank_matrices(self, l):
        center_x = (l[:,0]+1.) * self.center_x_factor
        center_y = (l[:,1]+1.) * self.center_y_factor
        delta = T.exp(l[:,2]) * self.delta_factor
        sigma = T.exp(l[:,3])
        gamma = T.exp(l[:,4]).dimshuffle(0, 'x')

        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*self.rngN
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*self.rngN
        
        FX = T.exp( -(self.a-muX.dimshuffle([0, 1, 'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FY = T.exp( -(self.b-muY.dimshuffle([0, 1, 'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + self.numtol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + self.numtol)

        return FY, FX, gamma

    def read(self, images, l):
        I = images.reshape((images.shape[0], self.img_height, self.img_width))
        FY, FX, gamma = self.filterbank_matrices(l)

        W = batched_dot(batched_dot(FY, I), FX.transpose([0,2,1])) 
        W = W.reshape((images.shape[0], self.N*self.N))
        W = gamma * W
        return W

    def write(self, windows, l):
        W = windows.reshape((windows.shape[0], self.N, self.N))
        FY, FX, gamma = self.filterbank_matrices(l)

        I = batched_dot(batched_dot(FY.transpose([0,2,1]), W), FX) 
        I = I.reshape((windows.shape[0], self.img_height*self.img_width))
        I = 1./gamma * I 
        return I
