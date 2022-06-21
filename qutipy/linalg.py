#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2022 Sumeet Khatri.
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

import numpy as np
from numpy.linalg import matrix_rank, norm

from qutipy.general_functions import dag,tensor,eye 
from qutipy.states import max_ent


def gram_schmidt(states, dim, normalize=True):
    """
    Performs the Gram-Schmidt orthogonalization procedure on the given states
    (or simply vectors). dim is the dimension of the vectors.
    """

    e = []
    u = []
    u.append(states[0])
    e.append(states[0] / norm(states[0]))

    for k in range(1, len(states)):
        S = np.array(np.zeros([dim, 1]), dtype=complex)
        for j in range(k):
            S += proj(u[j], states[k])
        u.append(states[k] - S)
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
    """

    [d1,d2]=X.shape

    gamma=max_ent(d2,normalized=False,density_matrix=False)

    return tensor(eye(d2),X)@gamma


def vec_inverse(v,d1,d2):
    """
    Take a bipartite vector v of dimension d1*d2, i.e., in the
    tensor product space C^(d1) ⊗ C^(d2), and transforms it 
    into a matrix of size d2 x d1.
    """

    gamma=max_ent(d1,normalized=False,density_matrix=False)

    return tensor(dag(gamma),eye(d2))@tensor(eye(d1),v)
