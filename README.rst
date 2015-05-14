Theano-Lights
=============

Theano-Lights is a research framework based on Theano providing implementation of several recent Deep learning models and a convenient training and test functionality. The models are not hidden behind layers of abstraction as in most deep learning platforms to enable more transparency and flexiblity during learning and research. 

Included models:
 * MLP with Dropout (FFN)
 * Convolutional neural network (CNN)
 * Variational Autoencoder  (VAE)
 * Convolutional Variational Autoencoder (CVAE)
 * Deep Recurrent Attentive Writer (DRAW)
 * LSTM Language model

Included features:
 * Batch normalization
 * Dropout
 * LSTM, GRU and SCRN recurrent layers 

Stochastic gradient descent variants:
 * SGD with momentum 
 * SGD Langevin dynamics
 * Rmsprop
 * Adam
 * Adam with gradient clipping

Supervised training on:
 * MNIST

Unsupervised training on:
 * MNIST
 * Frey Faces    
 * Penn Treebank
 * text8
