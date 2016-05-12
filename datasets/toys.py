# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from datasets.utils import shuffle_array
from sklearn.datasets import make_moons


# ============================================================================
#                   Moons
# ============================================================================

def load_moons(noise=0.05, n_samples=500, batchsize=32):
    """
    Load the Moon dataset using sklearn.datasets.make_moons() function.

    Params
    ------
        noise: (default=0.05) the noise of the moon data generator
        n_samples: (default=500) the total number of points generated
        batchsize: (default=32) the dataset batchsize
    
    Return
    ------
        source_data: dict with the separated data

    """
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=12345)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    X, y = shuffle_array(X, y)  # Usefull ?

    n_train = int(0.4*n_samples)
    n_val = int(0.3*n_samples)+n_train

    X_train, X_val, X_test = X[0:n_train], X[n_train:n_val], X[n_val:]
    y_train, y_val, y_test = y[0:n_train], y[n_train:n_val], y[n_val:]
    
    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data

# ============================================================================
#                   Clouds
# ============================================================================

def load_clouds(n_samples=50 ,n_classes=2, batchsize=10):
    """
    Dataset made from normal distributions localised at root of unity solution.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 training points 150 validation points 
            and 150 test points
        n_classes: (default=2) the number of classes
        batchsize: (default=10) the batchsize of the dataset
    
    Return
    ------
        source_data: dict with the separated data
    """
    # pos is the 2D positions as complex exponential numbers, root of unity solutions
    pos = [np.exp(2j*np.pi*i/n_classes) for i in range(n_classes)]

    X_train = np.empty((n_samples*n_classes, 2))
    X_train[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_train[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_train = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_val = np.empty((n_samples*n_classes, 2))
    X_val[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_val[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_val = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_test = np.empty((n_samples*n_classes, 2))
    X_test[:, 0] = np.hstack([np.random.normal(np.imag(p), 1/n_classes, size=n_samples) for p in pos])
    X_test[:, 1] = np.hstack([np.random.normal(np.real(p), 1/n_classes, size=n_samples) for p in pos])
    y_test = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_train, y_train = shuffle_array(X_train, y_train)
    X_val, y_val = shuffle_array(X_val, y_val)
    X_test, y_test = shuffle_array(X_test, y_test)

    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data

# ============================================================================
#                   X shaped data
# ============================================================================


def load_X(n_samples=50 ,n_classes=5, batchsize=20):
    """
    Dataset made from normal distributions and relocalised to make a X shape.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 training points 150 validation points 
            and 150 test points
        n_classes: (default=5) the number of classes
        batchsize: (default=20) the batchsize of the dataset
    
    Return
    ------
        source_data: dict with the separated data
    """
    def plouf(n_samples, n_classes, p):
        arr = np.random.normal(0, 1/n_classes, size=(n_samples, 2))
        arr = arr+np.sign(arr)*p
    return arr

    X_train = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_train = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_val = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_val = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_test = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_test = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_train, y_train = shuffle_array(X_train, y_train)
    X_val, y_val = shuffle_array(X_val, y_val)
    X_test, y_test = shuffle_array(X_test, y_test)

    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data



# ============================================================================
#                   Circles
# ============================================================================


def load_circles(n_samples=50 ,n_classes=5, batchsize=20):
    """
    Dataset made from normal distributions and relocalised to make a circles.

    Params
    ------
        n_samples: (default=50) the number of sample in each class and in each set.
            Example : 50 samples and 3 classes means 150 training points 150 validation points 
            and 150 test points
        n_classes: (default=5) the number of classes
        batchsize: (default=20) the batchsize of the dataset
    
    Return
    ------
        source_data: dict with the separated data
    """
    def plouf(n_samples, n_classes, p):
        arr = np.random.normal(0, 1/n_classes, size=(n_samples, 2))
        rho = np.sqrt(arr[:, 0]**2 + arr[:, 0]**2)
        phi = np.arctan2(arr[:, 1], arr[:, 0])
        rho = (p+1)/(1+np.exp(rho))/n_classes
        arr[:, 0] = rho * np.cos(phi)
        arr[:, 1] = rho * np.sin(phi)
        return arr

    X_train = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_train = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_val = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_val = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_test = np.vstack([plouf(n_samples, n_classes, c) for c in range(n_classes)])
    y_test = np.hstack([np.ones(n_samples)*i for i in range(n_classes)])

    X_train, y_train = shuffle_array(X_train, y_train)
    X_val, y_val = shuffle_array(X_val, y_val)
    X_test, y_test = shuffle_array(X_test, y_test)

    source_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'batchsize': batchsize,
                    }
    return source_data

