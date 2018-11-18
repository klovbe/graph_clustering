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
from keras.layers import Input, Dense, Lambda, Subtract, merge
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as K

from time import time
from datasets import *

from sklearn.cluster import KMeans
import metrics
from sklearn.manifold import TSNE

def get_encoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i], activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d, activation=activation_fn,
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])
    # Encoder model
    encoder = Model(input=x, output=y[K])
    return encoder

def get_decoder(node_num, d, K,
                n_units, nu1, nu2,
                activation_fn):
    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1],
                         activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=activation_fn,
                     W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])
    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(input=y, output=x_hat)
    return decoder

def get_autoencoder(encoder, decoder):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1],))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoencoder = Model(input=x, output=[x_hat, y])
    return autoencoder

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
    y = Dense(dims[0], activation='linear', kernel_initializer=init, kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
              name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

def autoencoder_bn(dims, act='relu', init='glorot_uniform'):
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
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = normalization.BatchNormalization()(h)
        h = Activation(activation=act)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], kernel_initializer=init, name='decoder_%d' % i)(y)
        y = normalization.BatchNormalization()(y)
        y = Activation(activation=act)(y)

    # output
    y = Dense(dims[0], activation='linear', kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        #self.clusters:(n_clusters, embedding dim), inputs:(None, embedding dim)
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # q:(None,n_clusters)
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Objectives
def weighted_mse_x(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    return K.sum(
        K.square(y_pred * y_true[:, :-1]),
        axis=-1) / y_true[:, -1]

def weighted_mse_y(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
    y_pred: Contains y2 - y1
    y_true: Contains s12
    '''
    min_batch_size = K.shape(y_true)[0]
    return K.reshape(
        K.sum(K.square(y_pred), axis=-1),
        [min_batch_size, 1]
    ) * y_true


class SDNE(object):

    def __init__(self, dims, bn, init, nu1, nu2, alpha, gamma, n_clusters, *hyper_dict, **kwargs):
        ''' Initialize the SDNE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
        '''

        self.alpha = alpha
        self.gamma = gamma
        self.dims = dims
        self.node_num = self.dims[0]
        self.n_stacks = len(self.dims) -1
        self.n_clusters = n_clusters

        # Generate encoder, decoder and autoencoder
        # If cannot use previous step information, initialize new models
        if bn:
            self.autoencoder, self.encoder = autoencoder_bn(self.dims, nu1=nu1, nu2=nu2, init=init)
        else:
            self.autoencoder, self.encoder = autoencoder(self.dims, nu1=nu1, nu2=nu2, init=init)

        # Initialize self.model
        # Input
        x_in = Input(shape=(2 * self.node_num,), name='x_in')
        x1 = Lambda(
            lambda x: x[:, 0:self.node_num],
            output_shape=(self.node_num,)
        )(x_in)
        x2 = Lambda(
            lambda x: x[:, self.node_num:2 * self.node_num],
            output_shape=(self.node_num,)
        )(x_in)
        # Process inputs
        [x_hat1, y1] = self.autoencoder(x1)
        [x_hat2, y2] = self.autoencoder(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])

        clustering_layer1 = ClusteringLayer(self.n_clusters, name='clustering')(y1)
        clustering_layer2 = ClusteringLayer(self.n_clusters, name='clustering')(y2)

        # Model
        # self.pre_model = Model(input=x_in, output=[x_diff1, x_diff2])
        self.pre_model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        self.cluster_model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff, clustering_layer1, clustering_layer2])

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256, gamma_s = 0.05):
        print('...Pretraining...')
        self.pre_model.compile(optimizer=optimizer, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y])
        # begin pretraining
        t0 = time()
        self.pre_model.fit_generator(generator=batch_generator_sdne(x, batch_size=batch_size, shuffle=True),
                                     epochs=epochs)
        print('Pretraining time: ', time() - t0)
        self.pretrained = True

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):

        sgd = SGD(lr=self.xeta, decay=1e-5, momentum=0.99, nesterov=True)
        # adam = Adam(lr=self.xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self.alpha]
        )

        self.model.fit_generator(
            generator=batch_generator_sdne(S, self.beta, self.n_batch, True),
            nb_epoch=self.num_iter,
            samples_per_epoch=S.nonzero()[0].shape[0] // self.n_batch,
            verbose=1
        )
        # Get embedding for all points
        self.Y = model_batch_predictor(self.autoencoder, S, self.n_batch)
        t2 = time()
        return self.Y, (t2 - t1)

    def fit(self, x_train, model_name, outdir, df_columns, optimizer=optimizer, y = None, epoch=500,
            batch_size=256, update_interval=5, early_stopping=20, tol=0.01):

        print('Update interval', update_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _, encoder_out = self.autoencoder.predict(x_train)
        y_pred = kmeans.fit_predict(encoder_out)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('kmeans : acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
            X_embedded = TSNE(n_components=2).fit_transform(encoder_out)
            plt.figure(figsize=(12, 10))
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
            plt.colorbar()
            plt.show()
        print(np.bincount(y_pred))

        # y_pred = kmeans.fit_predict(x_train)
        y_pred_last = np.copy(y_pred)
        self.cluster_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        self.cluster_model.compile(optimizer=optimizer, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y, 'kld', 'kld'],
                     loss_weights=[1, 1, args.alpha, args.gamma, args.gamma])
        # for ite in range(int(epoch)):
        #     if ite % update_interval == 0:
        #         q,_,_ = self.model.predict(x_train, verbose=0)
        #         p = self.target_distribution(q)  # update the auxiliary target distribution p
        #     y0 = np.zeros_like(x_train)
        #     self.model.fit(x=x_train, y=[p, y0, x_train], batch_size=batch_size)

        # Step 2: deep clustering
        for ite in range(int(epoch)):
            if ite % update_interval == 0:
                _, _, _, q, _ = self.cluster_model.predict(x_train, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                print("delta label:{}".format(delta_label))
                y_pred_last = np.copy(y_pred)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
                print(np.bincount(y_pred))
                if ite > update_interval and delta_label < tol:
                    print("Early stopping...")
                    break
            # train on batch
            self.cluster_model.fit_generator(
                generator=batch_generator_sdne(x_train, self.beta, self.n_batch, True),
                shuffle=False
            )

        print('training time: ', time() - t1)
        # save the trained model

        print("saving predict data...")
        _, encoder_out = self.autoencoder.predict(x_train)
        _, _, _, q, _ = self.cluster_model.predict(x_train, verbose=0)
        #k-means
        y_pred = kmeans.fit_predict(encoder_out)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('kmeans : acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
        print(np.bincount(y_pred))
        #this method
        y_pred = q.argmax(1)
        if y is not None:
            print("orginal cluster proportion: {}".format(np.bincount(y)))
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
            X_embedded = TSNE(n_components=2).fit_transform(encoder_out)
            plt.figure(figsize=(12, 10))
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
            plt.colorbar()
            plt.show()
        print(np.bincount(y_pred))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--gene_select', default=None, type=int)
    parser.add_argument('--gamma', default=0.1,type=float)
    parser.add_argument('--gamma_s', default=0.05, type=float)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument("--early_stopping", default=20, type=int)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--train_datapath", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--labelpath", default=None, type=str)
    parser.add_argument("--outDir", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--model_name", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--data_type", default="count", type=str)
    parser.add_argument("--metirc", default="pearson", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--trans', dest='trans', action='store_true')
    feature_parser.add_argument('--no-trans', dest='trans', action='store_false')
    parser.set_defaults(trans=True)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--bn', dest='bn', action='store_true')
    feature_parser.add_argument('--no-bn', dest='bn', action='store_false')
    parser.set_defaults(bn=True)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--gene_scale', dest='gene_scale', action='store_true')
    feature_parser.add_argument('--no-gene_scale', dest='gene_scale', action='store_false')
    parser.set_defaults(gene_scale=True)
    args = parser.parse_args()
    print(args)

    # load dataset
    edges, graph = load_newdata(args)

    init = 'glorot_uniform'
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.)

    # # prepare the DEC model
    Sdne = SDNE(dims=[edges,  300, 100, 30, 10],bn=args.bn)
    #
    if args.ae_weights is None:
        Sdne.pretrain(x=x_test, optimizer=optimizer,
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     gamma_s=args.gamma_s)
    else:
        Sdne.autoencoder.load_weights(args.ae_weights)
    #
    Sdne.model.summary()
    t0 = time()

    Sdne.fit(x_test, x_test, x_test, y=y, model_name=args.model_name, outdir=args.outDir, df_columns=df_columns,
             epoch=args.epoch, batch_size=args.batch_size,
                     update_interval=update_interval, early_stopping=args.early_stopping, tol=args.tol)