# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt


def plot_curve(stats, ax=None, label=None):
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

def plot_mat(mat, ax=None):
	"""
	Plot the given matrix.
	Useless function ?
	"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    sns.heatmap(mat, cmap=plt.cm.coolwarm, ax=ax)
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
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def plot_img_samples(datasets, n_sample=4, cmap='Greys_r'):
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
    rand = np.random.RandomState()
    for n in range(n_sample):
        i = rand.randint(source_data['X_test'].shape[0])
        for j, data in enumerate(datasets):
            sample = data['X_test'][i]
            ax = fig.add_subplot(n_sample, n_datasets, n*n_datasets+1+j)
            ax.axis('off')
            ax.imshow(sample, cmap=cmap)
            if 'name' in data:
                ax.set_title(data['name'])
    return fig, fig.get_axes()



def plot_learning_curve(stats, regex='acc', title=''):
	"""
	Plot the statistics from the given stats dictionary that contains the regex.

	Params
	------
		stats: the dictionary with the learning stats
		regex: (default='acc') the regex used to filter the stats
		title: (default='') the graph's title
	Return
	------

	"""
    keys = [k for k in stats.keys() if re.search(regex, k)]
    print(keys)
    fig, ax = plt.subplots()
    for k in keys:	
        # Plot learning accuracy curve
        ax.plot(final_stats[k], label=k)
    add_legend(ax, xlabel='epoch', ylabel='loss')
    fig.suptitle(title)
    return fig, ax
