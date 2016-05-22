# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from datasets.utils import shuffle_array


# ============================================================================
#                   Moons
# ============================================================================

def make_moons(noise=0.05, n_samples=500):
    """
    Load the Moon dataset using sklearn.datasets.make_moons() function.

    Params
    ------
        noise: (default=0.05) the noise of the moon data generator
        n_samples: (default=500) the total number of points generated
        batchsize: (default=32) the dataset batchsize

    Return
    ------
        X: The data (numpy array shape [n_samples, 2])
        y: The labels (numpy array shape [n_samples])
    """
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y

# ============================================================================
#                   Clouds
# ============================================================================

def make_clouds(n_samples=50, n_classes=2):
    """
    Dataset made from normal distributions localised at root of unity solution.

    Params
    ------
        n_samples: (default=50) the number of sample in each class.
            Example : 50 samples and 3 classes means 150 points
        n_classes: (default=2) the number of classes
        batchsize: (default=10) the batchsize of the dataset
    
    Return
    ------
        X: The data (numpy array shape [n_samples, 2])
        y: The labels (numpy array shape [n_samples])
    """
    # pos is the 2D positions as complex exponential numbers, root of unity solutions
    pos = [np.exp(2j*np.pi*i/n_classes) for i in range(n_classes)]

    X = np.empty((n_samples*n_classes, 2))
    X[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X, y = shuffle_array(X, y)
    return X, y

# ============================================================================
#                   X shaped data
# ============================================================================


def make_X(n_samples=50, n_classes=5):
    """
    Dataset made from normal distributions and relocalised to make a X shape.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 points
        n_classes: (default=5) the number of classes
        batchsize: (default=20) the batchsize of the dataset

    Return
    ------
        X: The data (numpy array shape [n_samples, 2])
        y: The labels (numpy array shape [n_samples])
    """
    def plouf(n_samples, n_classes, p):
        arr = np.random.normal(0, 1/n_classes, size=(n_samples, 2))
        arr = arr+np.sign(arr)*p/n_classes
        return arr

    X = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X, y = shuffle_array(X, y)
    return X, y


# ============================================================================
#                   Circles
# ============================================================================


def make_circles(n_samples=50, n_classes=5, n_dim=2, noise=1):
    """
    Dataset made from normal distributions and relocalised to make a circles.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 points
        n_classes: (default=5) the number of classes
        n_dim: (default=2) the dimension of the spheres
        noise: (default=1) the noise of the distance from origin
    
    Return
    ------
        X: The data (numpy array shape [n_samples, n_dim])
        y: The labels (numpy array shape [n_samples])
    """
    def plouf(n_samples, n_classes, n_dim, noise, p):
        X = np.random.normal(0, 1/n_classes, size=(n_samples, n_dim))
        norm = p+np.random.uniform(-noise*np.sqrt(1/n_classes), noise*np.sqrt(1/n_classes), size=(n_samples,1))/n_classes
        X = X*norm/np.sqrt(np.sum(X**2, axis=1)[:, None])
        return X

    X = np.vstack([plouf(n_samples, n_classes, n_dim, noise, c) for c in range(n_classes)])
    y = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X, y = shuffle_array(X, y)
    return X, y

