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
from scipy.linalg import expm

from qutipy.general_functions import Tr, dag, eye, ket, tensor
from qutipy.su import su_generators


def coherent_state_fermi(A, rep="JW", density_matrix=False):
    """
    Generates the fermionic coherent state vector for n modes, where A is a complex
    anti-symmetric n x n matrix. The matrix A should be at least 2 x 2 -- for one mode,
    the coherent state is the just the vacuum.

    The definition being used here comes from

        A. Perelomov. Generalized Coherent States and Their Applications (Sec. 9.2)

    """

    n = np.shape(A)[0]  # Number of modes

    a, _ = jordan_wigner(n)

    At = np.zeros((2**n, 2**n), dtype=complex)

    N = np.linalg.det(eye(n) + A @ dag(A)) ** (1 / 4)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            At = At + (-1 / 2) * A[i - 1, j - 1] * dag(a[j]) @ dag(a[i])

    vac = tensor([ket(2, 0), n])

    if not density_matrix:
        return (1 / N) * expm(At) @ vac
    else:
        coh = (1 / N) * expm(At) @ vac
        return coh @ dag(coh)


def cov_matrix_fermi(X, n, rep="JW"):
    """
    Generates the covariance matrix associated with the operator X. The underlying
    calculations are done using the specified representation, although the matrix
    itself is independent of the representation used for the calculation.
    """

    G = np.zeros((2 * n, 2 * n), dtype=complex)

    _, c = jordan_wigner(n)

    for j in range(1, 2 * n + 1):
        for k in range(1, 2 * n + 1):
            G[j - 1, k - 1] = (1j / 2) * Tr(X @ (c[j] @ c[k] - c[k] @ c[j]))

    return G


def jordan_wigner(n):
    """
    Generates the Jordan-Wigner representation of the fermionic creation, annihilation,
    and Majorana operators for an n-mode system.

    The convention for the Majorana operators is as follows:

        c_j=aj^{dag}+aj
        c_{n+j}=i(aj^{dag}-aj)

    """

    s = ket(2, 0) @ dag(ket(2, 1))

    S = su_generators(2)

    a = {}  # Dictionary for the annihilation operators
    c = {}  # Dictionary for the Majorana operators

    for j in range(1, n + 1):
        a[j] = tensor([S[3], j - 1], s, [S[0], n - j])
        c[j] = dag(a[j]) + a[j]
        c[n + j] = 1j * (dag(a[j]) - a[j])

    return a, c
