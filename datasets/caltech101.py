# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import tarfile
import re

import numpy as np
import matplotlib.image as mpimg
from datasets.utils import make_dataset, shuffle_array


np.random.seed(12345)

data_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_dir, 'data')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# ############### Download and prepare the Caltech 101 dataset ###############
# This is just some way of getting the Caltech 101 dataset from an online 
# location and loading it into numpy arrays. It doesn't involve Lasagne at all.


def _load_caltech(n_categories=10):
    """
    Load the raw Caltech dataset.
    Dataset full of images with 101 categories.

    Params
    ------
        n_categories: (default=10) the number of category
    
    Return
    ------
        source_data: dict with the separated data
    """
    filename = '101_ObjectCategories.tar.gz'
    if not os.path.exists(os.path.join(data_dir, filename)):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(data_dir, filename))

    with tarfile.open(os.path.join(data_dir, filename)) as tar:
        directories = [tarinfo for tarinfo in tar.getmembers() if tarinfo.isdir()]
        # Remove the root directory :
        directories = directories[1:]
        
        images = []
        categories = []
        categories_len = []
        if n_categories > len(directories):
            raise ValueError('The number of category ({}) must be less or equal to ({})'.format(n_categories, len(directories)))
        elif n_categories <= 0:
            n_categories = len(directories)

        print('Extracting images ...')
        for dd in directories[:n_categories]:
            print('\t extracting :', dd.name)
            regex = re.compile(dd.name+r'.*\.(jpg|png)$')
            catagory_name = os.path.basename(dd.name)
            categories.append(catagory_name)
            files = [tarinfo for tarinfo in tar.getmembers() 
                     if tarinfo.isreg() and regex.match(tarinfo.name)]
            categories_len.append(len(files))
            for file in files:
                img_file = tar.extractfile(file)
                img = mpimg.imread(img_file)
                img_file.close()
                images.append(img)
        return images, categories, categories_len


def _resize(images, shape=(100,100)):
    """
    Resize the given images to the given shape.

    Params
    ------
        images: the list of images
        shape: (default=(100,100)) the new images shape
    Return
    ------
        images_resize: the resized images (numpy array)
    """
    from scipy import misc
    images_resize = [misc.imresize(img, (100, 100)) for img in images]
    images_resize = [np.repeat(img, 3).reshape(img.shape+(3,)) if len(img.shape)<3 else img 
                     for img in images_resize]
    images_resize = np.array(images_resize)
    return images_resize


def load_caltech(n_categories=10, shape=(100,100), batchsize=20):
    """
    Dataset full of images with 101 categories.

    Params
    ------
        n_categories: (default=10) the number of category
        shape: (default=(100,100)) the shape of the images
        batchsize: (default=20) the batchsize of the dataset
    
    Return
    ------
        source_data: dict with the separated data
    """
    images, categories, categories_len = _load_caltech(n_categories=n_categories)
    images = _resize(images)
    y = np.hstack([np.ones(l)*i for i, l in enumerate(categories_len)])
    X, y  = shuffle_array(images, y)
    source_data = make_dataset(X, y, batchsize)
    source_data['categories'] = categories
    source_data['categories_len'] = categories_len
    return source_data
