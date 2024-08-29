#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2023 Sumeet Khatri.
#                       --.- ..- - .. .--. -.--
#
#
# SPDX-License-Identifier: AGPL-3.0
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import math

import numpy as np
from numpy.linalg import eig, matrix_rank, norm

from scipy.linalg import sqrtm

from qutipy.general_functions import dag, eye, tensor, ket
from qutipy.pauli import nQubit_Pauli_basis
#from qutipy.states import max_ent
from qutipy.su import nQudit_su_generators, su_generators
from qutipy.weyl import discrete_Weyl_basis, nQudit_discrete_Weyl_basis


def gamma(dim, as_matrix=True):
    """
    Generates the dim-dimensional *unnormalized) maximally entangled vector,
    which is defined as

    (|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If as_matrix=True, then the function returns the state as a density matrix.
    """

    Gamma = np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
    if as_matrix:
        return Gamma @ dag(Gamma)
    else:
        return Gamma
    

def gram_schmidt(vectors, dim, normalize=True):
    """
    Performs the Gram-Schmidt orthogonalization procedure on the given 
    vectors. dim is the dimension of the vectors.
    """

    e = []
    u = []
    u.append(vectors[0])
    e.append(vectors[0] / norm(vectors[0]))

    for k in range(1, len(vectors)):
        S = np.array(np.zeros([dim, 1]), dtype=complex)
        for j in range(k):
            S += proj(u[j], vectors[k])
        u.append(vectors[k] - S)
        e.append(u[k] / norm(u[k]))

    if normalize:
        return e
    else:
        return u


def proj(u, v):
    """
    Calculates the projection of vector v onto vector u.
    """

    return (complex(dag(u) @ v) / float(norm(u) ** 2)) * u


def rank(X):
    """
    Determines the rank of the matrix X, specified as a 2d numpy array.
    """

    return matrix_rank(X)


def vec(X):
    """
    Takes a matrix X of size d1 x d2 and 'vectorizes' it, by
    stacking the columns of X. The result is a bipartite vector
    of dimension d2*d1, i.e., in the tensor product space
    C^(d2) ⊗ C^(d1).

    TODO: Rewrite this function using numpy reshaping functions
    """

    [d1, d2] = X.shape

    Gamma = gamma(d2, as_matrix=False)

    return tensor(eye(d2), X) @ Gamma


def vec_inverse(v, d1, d2):
    """
    Take a bipartite vector v of dimension d1*d2, i.e., in the
    tensor product space C^(d1) ⊗ C^(d2), and transforms it
    into a matrix of size d2 x d1.
    """

    ### TODO: rewrite this function using numpy reshaping functions.

    Gamma = gamma(d1, as_matrix=False)

    return tensor(dag(Gamma), eye(d2)) @ tensor(eye(d1), v)


def generate_linear_op_basis(d, basis="w", local_dimension=2):
    """
    Generates a list of d^2 linear operators that span the space of
    all linear operators acting on a d-dimensional space.

    Choices for the basis are:

        - basis='w': the discrete-Weyl basis
        - basis='wtensor': the basis of tensor products of single-qudit
            discrete-Weyl operators. Valid when d=D^n.
        - basis='su': the SU(d) basis
        - basis='sutensor': the basis of tensor products of single-qudit
            discrete-Weyl operators. Valid when d=D^n.
        - basis='pauli': the basis of tensor products of single-qubit
            Pauli operators. Valid when d=2^n.

    """

    B = []

    if basis == "w":
        B = discrete_Weyl_basis(d)
        return B
    elif basis == "su":
        B = su_generators(d)
        return B
    elif basis == "pauli":
        if np.log2(d) - int(np.log2(d)) == 0:
            B = nQubit_Pauli_basis(int(np.log2(d)))
            return B
        else:
            return "The dimension must be an exponent of two!\n"
    elif basis == "wtensor":
        exponent = math.log(d, local_dimension)
        if exponent - int(exponent) == 0:
            B = nQudit_discrete_Weyl_basis(local_dimension, int(exponent))
        else:
            return "The dimension must be an exponent of the local dimension!\n"
    elif basis == "sutensor":
        exponent = math.log(d, local_dimension)
        if exponent - int(exponent) == 0:
            B = nQudit_su_generators(local_dimension, int(exponent))
        else:
            return "The dimension must be an exponent of the local dimension!\n"
    else:
        return "Improper basis choice!\n"


def eigenvalues(X):
    """
    Returns the eigenvalues of a square matrix X.
    """

    return eig(X)[0]


def eigenvectors(X):
    """
    Returns the eigenvectors of a square matrix X.
    """

    d = X.shape[0]

    E = eig(X)[0]  # The eigenvalues
    V = eig(X)[1]  # The eigenvectors

    # The eigenvectors are given by the columns of V, i.e., V[:,i].
    # We take these and reshape them to column vectors.

    v = [np.reshape(V[:, i], [d, 1]) for i in range(len(E))]

    return v


def eigensystem(X):
    """
    Returns the eigenvalues and eigenvectors of a square matrix X.
    """

    d = X.shape[0]

    E = eig(X)[0]  # The eigenvalues
    V = eig(X)[1]  # The eigenvectors

    # The eigenvectors are given by the columns of V, i.e., V[:,i].
    # We take these and reshape them to column vectors.

    v = [np.reshape(V[:, i], [d, 1]) for i in range(len(E))]

    return [(E[i], v[i]) for i in range(len(E))]


def Sqrtm(X):
    """
    Takes the matrix square root.

    We need this right now to set the datatype
    of the output of scipy's sqrtm to np.complex128.
    """

    return sqrtm(X).astype(np.complex128)


