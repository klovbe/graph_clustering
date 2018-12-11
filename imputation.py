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

    def __init__(self, dims1, dims2, bn, init, nu1, nu2, alpha, gamma, n_clusters, *hyper_dict, **kwargs):
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
        self.dims1 = dims1
        self.dims2 = dims2
        self.gene_num = self.dims1[0]
        self.cell_num = self.dims2[0]
        self.n_stacks1 = len(self.dims1) -1
        self.n_stacks2 = len(self.dims2) -1

        # Generate encoder, decoder and autoencoder
        # If cannot use previous step information, initialize new models
        if bn:
            self.autoencoder1, self.encoder1 = autoencoder_bn(self.dims1, nu1=nu1, nu2=nu2, init=init)
            self.autoencoder2, self.encoder2 = autoencoder(self.dims2, nu1=nu1, nu2=nu2, init=init)
        else:
            self.autoencoder1, self.encoder1 = autoencoder(self.dims1, nu1=nu1, nu2=nu2, init=init)
            self.autoencoder2, self.encoder2 = autoencoder(self.dims2, nu1=nu1, nu2=nu2, init=init)

        # Initialize self.model
        # Input
        x1 = Input(shape=(self.gene_num,), name='x1_in')
        x2 = Input(shape=(self.cell_num,), name='x2_in')
        # Process inputs
        x_hat1 = self.autoencoder1(x1)
        x_hat2 = self.autoencoder2(x2)
        y1 = self.encoder1(x1)
        y2 = self.encoder2(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])

        def cosine_distance(vests):
            x, y = vests
            # x = K.l2_normalize(x, axis=-1)
            # y = K.l2_normalize(y, axis=-1)
            return -K.mean(x * y, axis=-1, keepdims=True)

        def cos_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([y1, y2])
        # Model
        # self.pre_model = Model(input=x_in, output=[x_diff1, x_diff2])
        self.model = Model(input=[x1, x2], output=[x_diff1, x_diff2, y_diff])


    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256):
        print('...Pretraining...')
        self.model.compile(optimizer=optimizer, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y])
        # begin pretraining
        t0 = time()
        self.model.fit_generator(generator=batch_generator_sdne(x, batch_size=batch_size, shuffle=True, beta=beta),
                                     epochs=epochs)
        print('training time: ', time() - t0)


    # def fit(self, x, graph=None, edge_f=None,
    #                     is_weighted=False, no_python=False):
    #
    #     sgd = SGD(lr=self.xeta, decay=1e-5, momentum=0.99, nesterov=True)
    #     # adam = Adam(lr=self.xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #     self.model.compile(
    #         optimizer=sgd,
    #         loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
    #         loss_weights=[1, 1, self.alpha]
    #     )
    #
    #     self.model.fit_generator(
    #         generator=batch_generator_sdne(S, self.beta, self.n_batch, True),
    #         nb_epoch=self.num_iter,
    #         samples_per_epoch=S.nonzero()[0].shape[0] // self.n_batch,
    #         verbose=1
    #     )
    #     # Get embedding for all points
    #     self.Y = model_batch_predictor(self.autoencoder, S, self.n_batch)
    #     t2 = time()
    #     return self.Y, (t2 - t1)

    def fit(self, x_train, optimizer='adam', beta =1, y = None, epoch=500,
            batch_size=256, update_interval=5, early_stopping=20, tol=0.01):

        double_x = np.append(x_train, x_train, axis=1)
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
            # train on batch
            if ite % update_interval == 0:
                _, q = self.predict_model.predict(double_x, verbose=0)
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

            self.cluster_model.fit_generator(
                generator=batch_generator_sdne(x_train, batch_size=batch_size, shuffle=True, beta=beta),
                shuffle=False
            )

        print('training time: ', time() - t1)
        # save the trained model

        print("saving predict data...")
        _, encoder_out = self.autoencoder.predict(x_train)
        _, _, _, q, _ = self.cluster_model.predict(double_x, verbose=0)
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
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument("--early_stopping", default=20, type=int)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--train_datapath", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--labelpath", default=None, type=str)
    parser.add_argument("--outDir", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--model_name", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--data_type", default="count", type=str)
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
    parser.add_argument("--metirc", default="pearson", type=str)
    parser.add_argument('--gamma', default=0.1,type=float)
    parser.add_argument('--nu1', default=1e-6, type=float)
    parser.add_argument('--nu2', default=1e-6, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()
    print(args)

    # load dataset
    t0 = time()
    edges, graph = load_newdata(args)
    print("")
    y = None
    if args.labelpath is not None:
        from sklearn.preprocessing import LabelEncoder
        labeldf = pd.read_csv(args.labelpath, header=0, index_col=0)
        y = labeldf.values
        y = y.transpose()
        y = np.squeeze(y)
        if not isinstance(y, (int, float)):
            y = LabelEncoder().fit_transform(y)
        n_clusters = len(np.unique(y))
        # labeldf = pd.read_csv(args.labelpath, header=None, index_col=None)
        # y = labeldf.values
        # y = y.transpose()
        # y = np.squeeze(y)
        # n_clusters = len(np.unique(y))
        print("has {} clusters:".format(n_clusters))
        print("orginal cluster proportion: {}".format(np.bincount(y)))

    init = 'glorot_uniform'
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.)

    # # prepare the DEC model
    Sdne = SDNE(dims=[edges,  300, 100, 30, 10],bn=args.bn, init=init, nu1=args.nu1, nu2=args.nu2, alpha=args.alpha,
                gamma=args.gamma, n_clusters=n_clusters)
    Sdne.pre_model.summary()
    #
    if args.ae_weights is None:
        Sdne.pretrain(x=graph, optimizer=optimizer, epochs=args.pretrain_epochs, batch_size=args.batch_size)
    else:
        Sdne.autoencoder.load_weights(args.ae_weights)
    #
    Sdne.cluster_model.summary()

    Sdne.fit(graph, optimizer = optimizer, beta = 1, y=y,
             epoch=args.epoch, batch_size=args.batch_size,
                     update_interval=args.update_interval, early_stopping=args.early_stopping, tol=args.tol)