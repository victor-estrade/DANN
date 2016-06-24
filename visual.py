# -*- coding: utf-8 -*-
from __future__ import division, print_function

import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

CMAP = 'Paired'
# CMAP = 'Set1'


def curve(stats, ax=None, label=None):
    """
    Plot the given stats curve.
    Useless function ?
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Plot learning accuracy curve
    ax.plot(stats, label=label)
    return fig, ax


def mat(mat, ax=None):
    """
    Plot the given matrix.
    Useless function ?
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    sns.heatmap(mat, cmap=plt.cm.coolwarm, ax=ax)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return fig, ax


def add_legend(ax, xlabel='', ylabel='', title=''):
    """
    Add legend to the given axes.
    
    Params
    ------
        ax: axes on which to add a legend
        xlabel: (default='') the label of the x axis
        ylabel: (default='') the label of the y axis
        title: (default='') the title of the axes

    Return
    ------
        Nothing
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)


def img_samples(datasets, n_sample=4, cmap='Greys_r', random_state=None):
    """
    Plot images randomly sampled from the test set of given datasets.
    The sampling respect alignment. If the i-th image is sample from the first
    dataset then the other datasets will plot their i-th image.

    Params
    ------
        datasets: a list of dataset from which the images will be sampled
            (in the 'X_test' part)
        n_sample: (default=4) the number of images to be plotted
        cmap: (default='Greys_r') the cmap used to color the images
    Return
    ------
        fig: the figure
        axes: the axes array of the images

    """
    n_datasets = len(datasets)
    # Plot some sample images:
    fig = plt.figure()
    rand = np.random.RandomState(random_state)
    for n in range(n_sample):
        i = rand.randint(datasets[0]['X_test'].shape[0])
        for j, data in enumerate(datasets):
            sample = data['X_test'][i]
            ax = fig.add_subplot(n_sample, n_datasets, n*n_datasets+1+j)
            ax.axis('off')
            ax.imshow(sample, cmap=cmap)
            if 'name' in data:
                ax.set_title(data['name'])
    return fig, fig.get_axes()


def learning_curve(stats, regex='acc', title=''):
    """
    Plot the statistics from the given stats dictionary that contains the regex.

    Params
    ------
        stats: the dictionary with the learning stats
        regex: (default='acc') the regex used to filter the stats
        title: (default='') the graph's title
    Return
    ------
        fig: the figure (None if the regex match nothing)
        ax: the axes (None if the regex match nothing)
    """
    keys = [k for k in stats.keys() if re.search(regex, k)]
    if keys:
        fig, ax = plt.subplots()
        for k in keys:  
            # Plot learning accuracy curve
            ax.plot(stats[k], label=k)
        add_legend(ax, xlabel='epoch', ylabel=regex)
        fig.suptitle(title)
        return fig, ax
    else:
        return None, None


def bound(X, y, predict_fn, ax=None):
    """
    Plot the bounds of a 2D dataset (X,y) given a probability prediction
    function.
    
    Params
    ------
        X: the data
        y: the true labels
        predict_fn: the probability prediction function
        ax: (default=None) the axes
    Return
    ------
        fig: the figure
        ax: the axes
    """
    cm_heat = plt.cm.RdBu
    color = cm.ScalarMappable(cmap=CMAP)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)[0, :, 1]
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm_heat, alpha=.8)

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color)
    ax.xlim(xx.min(), xx.max())
    ax.ylim(yy.min(), yy.max())
    return fig, ax

# ============================================================================
#                   Data
# ============================================================================

def source_2D(X, y, ax=None):
    """
    Plot 2D data with the source color and shape settings
    
    Params
    ------
        X: the 2D data (numpy 2d-array)
        y: the labels of the data
        ax: (default=None)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    color = cm.ScalarMappable(cmap=CMAP)
    # y = y / np.linalg.norm(y)
    ax.scatter(X[:, 0], X[:, 1], label='source', marker='v', s=80, c=color.to_rgba(y))
    return fig, ax


def target_2D(X, y, ax=None):
    """
    Plot 2D data with the target color and shape settings
    
    Params
    ------
        X: the 2D data (numpy 2d-array)
        y: the labels of the data
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    color = cm.ScalarMappable(cmap=CMAP)
    # y = y / np.linalg.norm(y)
    ax.scatter(X[:, 0], X[:, 1], label='target', marker='o', s=80, c=color.to_rgba(y))
    return fig, ax


def corrected_2D(X, y, ax=None):
    """
    Plot 2D data with the corrected color and shape settings
    
    Params
    ------
        X: the 2D data (numpy 2d-array)
        y: the labels of the data
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    color = cm.ScalarMappable(cmap=CMAP)
    # y = y / np.linalg.norm(y)
    ax.scatter(X[:, 0], X[:, 1], label='corrected', marker='*', s=80, c=color.to_rgba(y))
    return fig, ax

# ============================================================================
#                   Clusters
# ============================================================================

def centers_source(C, ax=None):
    """
    Plot 2D data centers with the source color and shape settings
    
    Params
    ------
        C: the 2D data (numpy 2d-array)
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.scatter(C[:, 0], C[:, 1], label='source centers', marker='D', s=100, 
               edgecolors='purple', facecolors='purple')
    return fig, ax



def centers_target(C, ax=None):
    """
    Plot 2D data centers with the target color and shape settings
    
    Params
    ------
        C: the 2D data (numpy 2d-array)
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.scatter(C[:, 0], C[:, 1], label='target centers', marker='D', s=100, 
               edgecolors='yellow', facecolors='yellow')
    return fig, ax


def centers_corrected(C, ax=None):
    """
    Plot 2D data centers with the corrected color and shape settings
    
    Params
    ------
        C: the 2D data (numpy 2d-array)
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.scatter(C[:, 0], C[:, 1], label='corrected centers', marker='D', s=100, 
               edgecolors='green', facecolors='green')
    return fig, ax


def mapping(X, Y, ax=None):
    """
    Plot 2D data points mapping.
    
    Params
    ------
        X: the 2D data (numpy 2d-array), origin of the arrows
        Y: the 2D data (numpy 2d-array), target of the arrows
        ax: (default=None) the axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Plot clusters mapping to clusters (green)
    ax.quiver(X[:,0],X[:,1],
              Y[:,0]-X[:,0], Y[:,1]-X[:,1],
              scale_units='xy', angles='xy', scale=1, facecolors='g')
    return fig, ax

# ============================================================================
#                   Dimension reductions
# ============================================================================

def tsne(X, Y, y, ax=None, n_sample=100):
    """
    Plot data comparison throught a TSNE
    
    Params
    ------
        X: the 'source' data (numpy 2d-array)
        Y: the 'target' data (numpy 2d-array)
        y: the labels of both data
        ax: (default=None)
        n_sample: (default=100) int or float.
    """
    from sklearn.manifold import TSNE
    
    assert X.shape[0] == Y.shape[0] == y.shape[0], "Data should have the same number of sample"
    n = X.shape[0]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    if n_sample <= 0 or n_sample >= n:
        n_sample = n
    else:
        if 0 < n_sample < 1:
            n_sample = int(n_sample * n)
        else:
            n_sample = int(n_sample)
        idx = np.random.choice(n, size=n_sample, replace=False)
        X = X[idx]
        y_X = y[idx]
        Y = Y[idx]
        y_Y = y[idx]
    D = np.vstack((X, Y))
    ts = TSNE()
    Z = ts.fit_transform(D)
    n = X.shape[0]
    color = cm.ScalarMappable(cmap=CMAP)
    ax.scatter(Z[:n, 0], Z[:n, 1], label='X', marker='o', s=80, c=color.to_rgba(y_X))
    ax.scatter(Z[n:, 0], Z[n:, 1], label='Y', marker='*', s=80, c=color.to_rgba(y_Y))
    
    return fig, ax

