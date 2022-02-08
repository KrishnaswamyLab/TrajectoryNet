# -*- coding: utf-8 -*-
"""
===============================
1D Unbalanced optimal transport
===============================

This example illustrates the computation of Unbalanced Optimal transport
using a Kullback-Leibler relaxation.
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

##############################################################################
# Generate data
# -------------


#%% parameters

n = 2  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=1)  # m= mean, s= std
b = gauss(n, m=60, s=1)

a = [0.1, 0.9]
b = [0.9, 0.1]

# make distributions unbalanced
# b *= 5.

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()


##############################################################################
# Plot distributions and loss matrix
# ----------------------------------

#%% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.legend()

# plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, "Cost matrix M")


##############################################################################
# Solve Unbalanced Sinkhorn
# --------------


def get_transform_matrix(gamma, a, epsilon=1e-8):
    """Return matrix such that T @ a = b
    gamma : gamma @ 1 = a; gamma^T @ 1 = b
    """
    return (np.diag(1.0 / (a + epsilon)) @ gamma).T


def get_growth_coeffs(gamma, a, epsilon=1e-8, normalize=False):
    T = get_transform_matrix(gamma, a, epsilon)
    unnormalized_coeffs = np.sum(T, axis=0)
    if not normalize:
        return unnormalized_coeffs
    return unnormalized_coeffs / np.sum(unnormalized_coeffs) * len(unnormalized_coeffs)


# Sinkhorn

epsilon = 0.1  # entropy parameter
alpha = 1  # Unbalanced KL relaxation parameter
beta = 10000
# Gs = ot.emd(a, b, M)
Gs = sinkhorn_knopp_unbalanced(a, b, M, epsilon, alpha, beta, verbose=True)
print(Gs)
print(a, b)
print(get_growth_coeffs(Gs, np.array(a)))
print(get_transform_matrix(Gs, np.array(a)) @ a)
print(get_growth_coeffs(Gs, np.array(a)) * a)
exit()
print(Gs)
print(Gs @ np.ones_like(a))
print("bbbbbbbbbb")
tt = get_transform_matrix(Gs, np.array(a))
print(tt)
print("col_sum(tt)", np.sum(tt, axis=0))
print("tt @ a", tt @ a)
print("bbbbbbbbbb")
print(np.sum(Gs, axis=0), np.sum(Gs, axis=1))
print("aaaaaaa == 1")
alpha = 1  # Unbalanced KL relaxation parameter
Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha, verbose=True)
print(Gs)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, "UOT matrix Sinkhorn")

# pl.show()
