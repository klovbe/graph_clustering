disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import networkx as nx

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as K

from time import time
from datasets import *

from sklearn.cluster import KMeans
import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def autoencoder(dims, nu1, nu2, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                  name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
              name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                  name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], activation='sigmoid', kernel_initializer=init, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
              name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

def autoencoder_bn(dims, nu1, nu2, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(h)
        h = BatchNormalization()(h)
        h = Activation(activation=act)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1), kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(h)
    # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], kernel_initializer=init, name='decoder_%d' % i, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y)
        y = BatchNormalization()(y)
        y = Activation(activation=act)(y)

    # output
    y = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='decoder_0', kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')