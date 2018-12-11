import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.spatial.distance import *

def sizefactor(df):
    log = np.log(df+1)
    col_mean = np.mean(log, axis=0)
    col_mean = np.expand_dims(np.exp(col_mean), 0)
    div = np.divide(np.exp(log), col_mean)
    sf = np.median(div, axis=1)
    sf = np.expand_dims(sf, 1)
    div = np.log(np.divide(df, sf)+1)
    return div

def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div

def com_graph(df, metric):
    if metric == 'pearson':
        # graph = df.transpose().corr()
        graph = np.corrcoef(df.values)
    # If rowvar is True (default), then each row represents a variable, with observations in the columns.
    # Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
    elif metric == 'spearman':
        # graph = df.transpose().corr(method='spearman')
        graph, _ = spearmanr(df.values, axis=1)
        #If axis=0 (default), then each column represents a variable, with observations in the rows.
        # If axis=1, the relationship is transposed: each row represents a variable, while the columns contain observations.
        #  If axis=None, then both arrays will be raveled.
    elif metric == 'euclidean':
        graph = squareform(pdist(df.values, metric='euclidean'))
    elif metric == 'cosine':
        graph = 1-squareform(pdist(df.values, metric='cosine'))
    else:
        raise IOError('undefined metric')
    return graph

def load_newdata(args):
    print("make dataset from {}...".format(args.train_datapath))
    df = pd.read_csv(args.train_datapath, sep=",", index_col=0)
    if args.trans:
        df = df.transpose()
    print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
    # o_len = len(df)
    # x = 0
    # index_list = []
    # i_index_list =[]
    # for i, row in enumerate(df.index):
    #   if (df.loc[row, :] > 0.0).sum() > x:
    #     index_list.append(row)
    #     i_index_list.append(i)
    # df = df.loc[index_list, :]
    # if len(df) < o_len:
    #   i_df = pd.DataFrame(data=i_index_list)
    #   i_df.to_csv("{}_filter_label.csv".format(FLAGS.model_name))
    # # print(df.shape)
    # y = 0
    # col_list = []
    # for col in df.columns:
    #   if (df[col] > 0.0).sum() > y:
    #     col_list.append(col)
    # df = df[col_list]
    # print("after filtering, have {} samples, {} features".format(df.shape[0], df.shape[1]))
    if args.data_type == 'count':
        df = row_normal(df)
        # df = sizefactor(df)
    elif args.data_type == 'rpkm':
        df = np.log(df + 1)
    if args.gene_select is not None:
        selected = np.std(df, axis=0)
        selected =selected.argsort()[-args.gene_select:][::-1]
        df = df[selected.index]
    if args.gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns)
    edges, _ = df.shape
    graph = com_graph(df, args.metric)
    return edges, graph

def batch_generator_sdne(X, batch_size, shuffle, beta=1):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :]
        X_batch_v_j = X[col_indices[batch_index], :]
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X_batch_v_i != 0, 1).reshape((batch_size, 1))
        deg_j = np.sum(X_batch_v_j != 0, 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generator_dec(X, p, batch_size, shuffle, beta=1):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :]
        X_batch_v_j = X[col_indices[batch_index], :]
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X_batch_v_i != 0, 1).reshape((batch_size, 1))
        deg_j = np.sum(X_batch_v_j != 0, 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        p1 = p[row_indices[batch_index], :]
        p2 = p[col_indices[batch_index], :]
        OutData = [a1, a2, X_ij.T, p1, p2]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def model_batch_predictor(model, X, batch_size):
    n_samples = X.shape[0]
    counter = 0
    pred = None
    while counter < n_samples // batch_size:
        _, curr_pred = \
            model.predict(X[batch_size * counter:batch_size * (counter + 1),
                          :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
        counter += 1
    if n_samples % batch_size != 0:
        _, curr_pred = \
            model.predict(X[batch_size * counter:, :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
    return pred

