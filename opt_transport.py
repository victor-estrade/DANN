#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

########### Compute transport with the Sinkhorn algorithm
## ref "Sinkhorn distances: Lightspeed computation of Optimal Transport", NIPS 2013, Marco Cuturi

def computeTransportSinkhorn(w_S, w_T, M, reg, max_iter=200, epsilon=1e-4):
    """
    Optimal transport solver. Compute transport with the Sinkhorn algorithm
    
    ref "Sinkhorn distances: Lightspeed computation of Optimal Transport", NIPS 2013, Marco Cuturi
    
    Params
    ------
        w_S: (numpy array [n_S]) mass of the source distribution (histogram)
        w_T: (numpy array [n_T]) mass of the target distribution (histogram)
        M: (numpy array [n_s, n_T]) cost matrix, 
            m_ij = cost to get mass from source point x_i to target point x_j
        reg: (float) lambda, value of the lagrange multiplier handling the entropy constraint
    Return
    ------
        transp : the transport matrix
    """
    # init data
    # ---------
    Nini = len(w_S)
    Nfin = len(w_T)
    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones(Nini)/Nini
    uprev = np.zeros(Nini)
    K = np.exp(-reg*M)  # Central matrix
    cpt = 0
    err = 1
    # Main loop
    # ---------
    while (err > epsilon and cpt < max_iter):
        cpt = cpt +1
        # First we do a sanity check
        if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Infinity')
            if cpt!=0:
                u = uprev
            break
        uprev = u  # Save the previous results in case of divide by 0
        # now the real algo part : update vectors u and v
        v = w_T/np.dot(K.T,u)
        u = w_S/np.dot(K,v)
        # Computing the new error value
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the n-th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-w_T))**2
    # End of Main loop
    # Return the transpotation matrix
    return u[:, np.newaxis]*K*v[:, np.newaxis].T


def computeTransportSinkhornLabelsLpL1(distribS,LabelsS, distribT, M, reg,
    p = 0.5, max_iter=200, eta=0.1, epsilon=1e-4):
    """
    Optimal transport solver.

    Params
    ------
        distribS: (numpy array [n_S]) mass of the source distribution (histogram)
        LabelsS : 
        distribT: (numpy array [n_T]) mass of the target distribution (histogram)
        M: (numpy array [n_s, n_T]) cost matrix, 
            m_ij = cost to get mass from source point x_i to target point x_j
        reg: (float) lambda, value of the lagrange multiplier handling the entropy constraint
    Return
    ------
        transp : the transport matrix
    
    ref
    ---
        N. Courty, R. Flamary, and D. Tuia, 
        “Domain adaptation with regularized optimal transport,”
        Lect. Notes Comput. Sci. 
        (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics),
        vol. 8724 LNAI, no. PART 1, pp. 274–289, 2014.
    
    """
    # init data
    indices_labels = [np.where(LabelsS==c)[0] for c in np.unique(LabelsS)]
    # Previous suboptimal version :
#     idx_begin = int(np.min(LabelsS))
#     for c in range(idx_begin,int(np.max(LabelsS)+1)):
#         idxc = indices(LabelsS, lambda x: x==c)
#         indices_labels.append(idxc)
    W = np.zeros(M.shape)
    # Majoration - Minimization process :
    # -----------------------------------
    for _ in range(10):
        Mreg = M + eta*W
        transp = computeTransportSinkhorn(distribS, distribT, Mreg, reg, max_iter=max_iter)
        # the transport has been computed. Check if classes are really separated
        for idx in indices_labels:
            W[idx, :] = p*((np.sum(transp[idx], 0)[np.newaxis, :]+epsilon)**(p-1))
        # Previous suboptimal version :
#         W = np.ones((Nini,Nfin))
#         for t in range(Nfin):
#             column = transp[:,t]
#             for c in range(len(indices_labels)):
#                 col_c = column[indices_labels[c]]
#                 W[indices_labels[c],t]=(p*((sum(col_c)+epsilon)**(p-1)))
    return transp


def opt_transp_sup(X_src, X_tgt, y_src=None, y_tgt=None, reg=10, max_iter=200, epsilon=1e-5, p=0.5):
    """
    Optimal transport solver.

    Params
    ------
        TODO
    Return
    ------
        transp : the transport matrix

    """
    if y_tgt is None:
        # Compute weights/mass/histograms
        K1rbf = rbf_kernel(X_src, X_src, 2)
        w_S = np.sum(K1rbf,1) / np.sum(K1rbf)
        K2rbf = rbf_kernel(X_tgt, X_tgt, 2)
        w_T = np.sum(K2rbf,1) / np.sum(K2rbf)

        # Compute cost matrix
        M = euclidean_distances(X_src, X_tgt)

        # Compute transport
        if y_src is None:
            transp = computeTransportSinkhorn(w_S, w_T, M, reg, max_iter=200, epsilon=1e-5)
        else:
            transp = computeTransportSinkhornLabelsLpL1(w_S, y_src, w_T, M, reg, max_iter=200, epsilon=1e-5, p=p)
        return transp
    elif y_src is None:
        raise ValueError('y_src must be given if y_tgt is provided')
    else:
        indexes = [(np.where(y_src == label)[0], np.where(y_tgt == label)[0]) for label in np.unique(y_src)]
        l = []
        transp = np.zeros((X_src.shape[0], X_tgt.shape[0]))
        for idx_src, idx_tgt in indexes:
            X_s = X_src[idx_src]
            X_t = X_tgt[idx_tgt]
            # Compute weights/mass/histograms
            K1rbf = rbf_kernel(X_s, X_s, 2)
            w_S = np.sum(K1rbf,1) / np.sum(K1rbf)
            K2rbf = rbf_kernel(X_t, X_t, 2)
            w_T = np.sum(K2rbf,1) / np.sum(K2rbf)

            # Compute cost matrix
            M = euclidean_distances(X_s, X_t)

            # Compute transport
            s_transp = computeTransportSinkhorn(w_S, w_T, M, reg, max_iter=200, epsilon=1e-5)

            # TODO : find another way to compute these next 2 ugly loops
            for i, i1 in enumerate(idx_src):
                for j, j1 in enumerate(idx_tgt):
                    transp[i1,j1] = s_transp[i,j]
        return transp
