# -*- coding: utf-8 -*-
from __future__ import division, print_function

import theano
import lasagne

import theano.tensor as T
import numpy as np

from decorators import batchpad
from sklearn.cluster import KMeans


def shuffle_array(*args):
    """
    Shuffle the given data. Keeps the relative associations arr_j[i] <-> arr_k[i].

    Params
    ------
        args: (numpy arrays tuple) arr_1, arr_2, ..., arr_n to be shuffled.
    Return
    ------
        X, y : the shuffled arrays.
    """
    # Assert that there is at least one array
    if len(args) == 0:
        raise ValueError('shuffle_array() must take at least one argument')
    length = args[0].shape[0]
    # Assert that every array have the same 1st dimension length:
    for i, arr in enumerate(args):
        assert arr.shape[0] == length, "Every array should have the same shape: " \
                        " array {} length = {}  array 1 length = {} ".format(i+1, arr.shape[0], length)
    # Make the random indices
    indices = np.arange(length)
    np.random.shuffle(indices)
    # Return shuffled arrays
    return tuple(arr[indices] for arr in args)


# ============================================================================
#                   Classwise random shuffle
# ============================================================================


def _classwise_shuffle(X, y):
    """
    Shuffle X without changing the class positions

    Params
    ------
        X: the data (numpy array)
        y: the labels 
    Return
    ------
        X_shuffled: Shuffled X without changing the class matching
    """
    idx = np.empty_like(y, dtype=int)
    for label in np.unique(y):
        arr = np.where(y==label)[0]
        arr2 = np.random.permutation(arr)
        idx[arr] = arr2
    return X[idx]


def classwise_shuffle(data, trainer, epoch, *args, **kwargs):
    """
    Randomly align the data according to their labels.
    
    Note : applied on data['X_train'] according to data['labels']

    The signature is the one used in the epoch preprocessing.

    """
    data['X_train'] = _classwise_shuffle(data['X_train'], data['labels'])
    return None


# ============================================================================
#                   K-closest data points
# ============================================================================


# http://stackoverflow.com/questions/25886374/pdist-for-theano-tensor
# Tested and approved
X = T.fmatrix('X')
Y = T.fmatrix('Y')
translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
euclidiean_distances = (translation_vectors ** 2).sum(2)
_euclidean_distance = theano.function([X, Y], euclidiean_distances, allow_input_downcast=True)
_euclidean_distance.__doc__="""
    Computes the distances between all pairs of row vectors in X and Y, the given matrices.
    Use GPU.
    Params
    ------
        X: the fist matrix of row vectors
        Y: the second matrix of row vectors
    Return
    ------
        M_dist: the distance matrix
"""


def kclosest(X, Y, k, batchsize=None):
    """
    Computes for each sample from X the k-closest samples in Y and return 
    their index.

    Params
    ------
        X: (numpy array [n_sample, n_feature])
        Y: (numpy array [n_sample, n_feature])
        k: (int)
    Return
    ------
        kclosest : (numpy array [n_sample, k]) the ordered index of 
            the k-closest instances from Y to X samples
    """
    assert X.shape == Y.shape
    N = X.shape[0]
    if batchsize is None:
        dist = _euclidean_distance(X.reshape(N, -1), Y.reshape(N, -1))
    else:
        dist = np.empty((N, N), dtype=theano.config.floatX)
        batch = np.arange(0, N+batchsize, batchsize)
        for excerpt_X in (slice(i0, i1) for i0, i1 in zip(batch[:-1], batch[1:])):
            dist[excerpt_X] = _euclidean_distance(X[excerpt_X], Y)

    if k > 0 and k <= N:
        kbest = np.argsort(dist, axis=1)[:, :k]
    else:
        kbest = np.argsort(dist, axis=1)

    return kbest


def clostest_map(X, Y, y, k=20, batchsize=None, shuffle=False):
    """
    Get the k-closest between X and Y restricted to the points 
    that have the same label in y.

    And for each point from X, maps it with the k-closest 
    that have been the less chosen yet.

    return the map as an index array to realign the X array.
    """
    assert X.shape[0] == Y.shape[0] == y.shape[0], \
        "The given datas and labels should have the same number of examples"\
        "(X {} == Y {} == y {})".format(X.shape[0], Y.shape[0], y.shape[0])
    counter = np.zeros(X.shape[0], dtype=int)
    idx = np.empty_like(y, dtype=int)
    for label in np.unique(y):
        # Get the examples of the right label
        idx_label = np.where(y==label)[0]

        # Get the k-closest index from the small part with the same labels
        idx_label2 = kclosest(X[idx_label], Y[idx_label], k, batchsize=batchsize)

        # Then we shuffle the arrays to prevent deterministic 
        # problems in the choose of the nearest point
        if shuffle:
            idx_label, idx_label2 = shuffle_array(idx_label, idx_label2)
        
        for i1, i2 in zip(idx_label, idx_label2):
            # i2 is an index array of shape (k,) with the sorted closest example index 
            # (of the sorted single class array)
            # Then idx_label[i2] are the sorted original index of the k-closest examples
            i = idx_label[i2[np.argmin(counter[idx_label[i2]])]]
            # i contains the chosen one, in the (k-)clostest example, with the minimum counter
            counter[i] = counter[i]+1
            idx[i1] = i
    return idx


# Méthode bourin. K-clostest on every data point
def exhaustive_clostest(data, trainer, epoch, *args, **kwargs):
    X = data['X_train']
    k = data['k'] if 'k' in data else 5

    @batchpad(data['batchsize'], X.shape, X.dtype)
    def f_output(X, trainer):
        return trainer.output(X)[0]
    
    X_out = f_output(X, trainer)
    X_trg = data['y_train']
    labels = data['labels']
    idx = clostest_map(X_out, X_trg, labels, k=k, batchsize=None)
    data['X_train'] = X[idx]
    return idx


# ============================================================================
#                   K-closest data clusters
# ============================================================================

def build_clusters(X, y, n_clusters=10):
    """

    Example
    -------
    >>> X = corrector_data['X_train']
    >>> y = source_data['y_train']
    >>>
    >>> centers_array, clusters_label, centers_labels = build_clusters(X, y, 5)
    >>>
    """
    classes = np.unique(y)
    shape = X.shape
    # Build the clusters for target data
    centers = []
    clusters_label = np.empty(X.shape[0], dtype=int)
    labels = []
    for label in classes:
        km = KMeans(n_clusters=n_clusters)
        # get the indexes of the data from the same category
        idx = np.where(y==label)[0]
        X_ = X[idx]
        km.fit(X_.reshape((X_.shape[0], -1)))
        centers.append(km.cluster_centers_.reshape((n_clusters,)+shape[1:]))
        clusters_label[idx] = km.labels_+label*n_clusters
        labels.append(np.ones(n_clusters, dtype=int)*label)
    
    centers_array = np.vstack(centers)
    centers_labels = np.hstack(labels)
    return centers_array, clusters_label, centers_labels



def cluster_preprocess(data, trainer, epoch, *args, **kwargs):
    k = data['k'] if 'k' in data else -1
    
    @batchpad(data['batchsize'], None, X.dtype)
    def f_output(X, trainer):
        return trainer.output(X)[0]
    
    X_out = f_output(data['X_train_centers'], trainer)
    X_trg = data['y_train_centers']
    labels = data['centers_labels']
    
    idx = clostest_map(X_out, X_trg, labels, k=k, batchsize=None)
    # idx takes the indexes to relabel the closest clusters
    # [2, 0, 1, 3]  means  0<-2, 1<-0, 2<-1, 3<-3
    data['X_train_closest_cluster'] = idx[data['X_train_clusters'][:]]
    # Now data['X_train_closest_cluster'] contains the nearest clusters label
    # from each data in X_train to the clusters of y_train
    
    # Do a random realign from here
    data['X_train'] = _classwise_shuffle(data['X_train'], data['X_train_closest_cluster'])
    
    # Or a fully k-closest realignment
    #idx = clostest_map(f_output(data['X_train'], trainer), data['y_train'],
    #                            data['X_train_closest_cluster'], k=k, batchsize=None)
    #data['X_train'] = data['X_train'][idx]
    data['preprocess'] = idx
    return idx
    