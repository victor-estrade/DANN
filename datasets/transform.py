# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from datasets.utils import Dataset

from sklearn.preprocessing import Normalizer
np.random.seed(12345)

# ============================================================================
#                   Matrix products
# ============================================================================

def _diag_dominant_matrix(size, safe=False):
    """
    https://matthewhr.wordpress.com/2013/09/01/how-to-construct-an-invertible-matrix-just-choose-large-diagonals/
    """
    matrix = np.random.random((size, size))
    s = np.sum(matrix, axis=1)
    matrix += np.eye(size) * s
    if safe:
        eigens = np.linalg.eig(matrix)[0]
        if (eigens == 0).any():
            raise ValueError('The matrix is not invertible. Internet lied to me ! {}'.format(str(eigens)))
    return matrix


def diag_dominant(X, y, normalize=False):
    """
    Transform the given dataset by applying a diagonal dominant matrix to it.

    target_data <- source_data . Matrix
    
    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """
    size = np.prod(X.shape[1:])
    y_t = np.copy(y)
    
    A = _diag_dominant_matrix(size)
    if normalize:
        normalizer = Normalizer()
        X_t = normalizer.fit_transform(np.dot(X.reshape(-1, size), A)).reshape(X.shape)
    else:
        X_t = np.dot(X.reshape(-1, size), A).reshape(X.shape)
        
    return X_t, y_t


def random_mat(X, y, normalize=False, random_state=None):
    """
    Transform the given dataset by applying a random matrix to it.

    target_data <- source_data . Matrix
    
    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """

    size = np.prod(X.shape[1:])
    y_t = np.copy(y)

    rand_state = np.random.RandomState(random_state)
    A = rand_state.rand(size, size)
    if normalize:
        normalizer = Normalizer()
        X_t = normalizer.fit_transform(np.dot(X.reshape(-1, size), A)).reshape(X.shape)
    else:
        X_t = np.dot(X.reshape(-1, size), A).reshape(X.shape)
    return X_t, y_t

# ============================================================================
#                   Rotations
# ============================================================================

def _rot_mat(angle):
    theta = (angle/180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    return rot_matrix

 
def _rotate_data(X, angle=35.):
    """Apply a rotation on a 2D dataset.
    """
    theta = (angle/180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta),  np.cos(theta)]])
    X_r = np.empty_like(X)
    X_r[:] = X[:].dot(rot_matrix)
    return X_r


def rotate(X, y, angle=35.):
    """
    Transform the given dataset by applying a rotation to it.

    target_data <- source_data . Rotation_Matrix

    Can be used only on 2D datasets !

    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """

    y_t = np.copy(y)
    X_t =  _rotate_data(X, angle=angle)
    return X_t, y_t


# ============================================================================
#                   Invertions
# ============================================================================

def invert(X, y, pivot=1):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- (pivot - source_data)
    
    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """
    y_t = np.copy(y)
    X_t = pivot - X
    return X_t, y_t

    
# ============================================================================
#                   Permutations
# ============================================================================

def mirror(X, y, shape=(-1,28,28)):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- np.fliplr(source_data.reshape(shape))

    TODO : This function works only with MNIST for now !!!
    
    Params
    ------
        X: The data (numpy array of shape [n_samples, fliped_dim, etc])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, fliped_dim, etc])
        y_t: the labels (deep copy of y)
    """

    X_t =  np.fliplr(np.copy(X).reshape(shape))
    y_t = np.copy(y)
    return X_t, y_t


def random_permut(X, y, random_state=None):
    """
    Transform the given dataset by applying a simple operation to it.

    target_data <- random_permutation of feature(source_data)

    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """
    y_t = np.copy(y)

    X_t = np.copy(X)
    # Take care of 3D or n-D data
    rand_state = np.random.RandomState(random_state)
    permutation = rand_state.permutation(np.prod(X_t.shape[1:]))
    
    shape = X_t.shape
    X_t = X_t.reshape(-1, np.prod(shape[1:]))
    X_t = X_t[:, permutation].reshape(shape)
    return X_t, y_t
    
# ============================================================================
#                   Grid Bend
# ============================================================================

    
def grid_bend(X, y, nx=10, ny=10, noise=0.3, grid=None):
    """
    Transform the given dataset by linear interpolation of a noisy grid.

    Params
    ------
        X: The data (numpy array of shape [n_samples, n_features])
        y: the labels (unused)

    Return
    ------
        X_t: the transformed data (numpy array of shape [n_samples, n_features])
        y_t: the labels (deep copy of y)
    """
    def barycentric_coords(r, r1, r2, r3):
        """
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        """
        x, y = r
        x1, y1 = r1
        x2, y2 = r2
        x3, y3 = r3
        det_T = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
        lambda_1 = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/det_T
        lambda_2 = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/det_T
        lambda_3 = 1-lambda_1-lambda_2
        return lambda_1, lambda_2, lambda_3

    if grid is None:
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=nx),
                             np.linspace(y_min, y_max, num=ny))
        grid = Dataset(xx=xx, yy=yy, nx=nx, ny=ny)
    # If the grid have
    if 'xxx' not in grid:
        grid.xxx = grid.xx + np.random.randn(*grid.xx.shape)*noise*(x_max-x_min)/nx
    if 'yyy' not in grid:
        grid.yyy = grid.yy + np.random.randn(*grid.yy.shape)*noise*(y_max-y_min)/ny

    #
    news = []
    for x in X:
        # Find the triangle
        xx = grid.xx
        yy = grid.yy
        for i in range(xx.shape[1]-1):
            if x[0] >= xx[0, i] and x[0] <= xx[0, i+1]:
                break
        for j in range(yy.shape[0]-1):
            if x[1] >= yy[j, 0] and x[1] <= yy[j+1, 0]:
                break
        # The triangle is :
        r1 = np.array([grid.xx[j,i], grid.yy[j,i]])
        r2 = np.array([grid.xx[j,i+1], grid.yy[j+1,i]])
        r3 = np.array([grid.xx[j,i+1], grid.yy[j,i]])
        # get the barycentric coords :
        lambda_1, lambda_2, lambda_3 = barycentric_coords(x, r1, r2, r3)
        # If it is OK then get the new position of the triangle vertices
        if 0 <= lambda_1 <= 1 and 0 <= lambda_2 <= 1 and 0 <= lambda_3 <= 1:
            r1 = np.array([grid.xxx[j,i], grid.yyy[j,i]])
            r2 = np.array([grid.xxx[j,i+1], grid.yyy[j+1,i]])
            r3 = np.array([grid.xxx[j,i+1], grid.yyy[j,i]])            
        else:  # The point is ouside of the triangle. Try the other one
            r3 = np.array([grid.xx[j,i], grid.yy[j+1,i]])
            lambda_1, lambda_2, lambda_3 = barycentric_coords(x, r1, r2, r3)
            r1 = np.array([grid.xxx[j,i], grid.yyy[j,i]])
            r2 = np.array([grid.xxx[j,i+1], grid.yyy[j+1,i]])
            r3 = np.array([grid.xxx[j,i], grid.yyy[j+1,i]])            
        # Add the new position of the data point.
        news.append(lambda_1*r1 + lambda_2*r2 + lambda_3*r3)
    
    # Gather the results and return it
    X_t = np.array(news)
    y_t = np.copy(y)
    return X_t, y_t, grid
