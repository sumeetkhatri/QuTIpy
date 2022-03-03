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

import itertools

import numpy as np
from numpy.linalg import matrix_power

from qutipy.general_functions import (  # NOTE: What does the tensor function does?
    Tr,
    dag,
    ket,
    tensor,
)


def discrete_Weyl_X(d):
    """
    Generates the X shift operators.
    """

    X = ket(d, 1) @ dag(ket(d, 0))

    for i in range(1, d):
        X = X + ket(d, (i + 1) % d) @ dag(ket(d, i))

    return X


def discrete_Weyl_Z(d):
    """
    Generates the Z phase operators.
    """

    w = np.exp(2 * np.pi * 1j / d)

    Z = ket(d, 0) @ dag(ket(d, 0))

    for i in range(1, d):
        Z = Z + w**i * ket(d, i) @ dag(ket(d, i))

    return Z


def discrete_Weyl(d, a, b):
    """
    Generates the discrete Weyl operator X^aZ^b.
    """

    return matrix_power(discrete_Weyl_X(d), a) @ matrix_power(discrete_Weyl_Z(d), b)


def generate_nQudit_X(d, indices):
    """
    Generates a tensor product of discrete Weyl-X operators. indices is a
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    """

    X = discrete_Weyl_X(d)

    out = 1

    for index in indices:
        out = tensor(out, matrix_power(X, index))

    return out


def generate_nQudit_Z(d, indices):
    """
    Generates a tensor product of discrete Weyl-Z operators. indices is a
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    """

    Z = discrete_Weyl_Z(d)

    out = 1

    for index in indices:
        out = tensor(out, matrix_power(Z, index))

    return out


def nQudit_cov_matrix(X, d, n):
    """
    Generates the matrix of second moments (aka covariance matrix) of an
    n-qudit operator X.
    """

    S = nQudit_quadratures(d, n)

    V = np.array(np.zeros((2 * n, 2 * n)), dtype=np.complex128)

    for i in range(2 * n):
        for j in range(2 * n):
            V[i, j] = Tr(X @ S[i + 1] @ dag(S[j + 1]))

    return V


def nQudit_quadratures(d, n):
    """
    Returns the list of n-qudit "quadrature" operators, which are defined as
    (for two qudits)

        S[0]=X(0) ⊗ Id
        S[1]=Z(0) ⊗ Id
        S[2]=Id ⊗ X(0)
        S[3]=Id ⊗ Z(0)

    In general, for n qubits:

        S[0]=X(0) ⊗ Id ⊗ ... ⊗ Id
        S[1]=Z(0) ⊗ Id ⊗ ... ⊗ Id
        S[2]=Id ⊗ X(0) ⊗ ... ⊗ Id
        S[3]=Id ⊗ Z(0) ⊗ ... ⊗ Id
        .
        .
        .
        S[2n-2]=Id ⊗ Id ⊗ ... ⊗ X(0)
        S[2n-1]=Id ⊗ Id ⊗ ... ⊗ Z(0)
    """

    S = {}

    count = 0

    for i in range(1, 2 * n + 1, 2):
        v = list(np.array(dag(ket(n, count)), dtype=int).flatten())
        S[i] = generate_nQudit_X(d, v)
        S[i + 1] = generate_nQudit_Z(d, v)
        count += 1

    return S


def nQudit_Weyl_coeff(X, d, n):
    """
    Generates the coefficients of the operator X acting on n qudit
    systems.
    """

    C = {}

    S = list(itertools.product(*[range(0, d)] * n))

    for s in S:
        s = list(s)
        for t in S:
            t = list(t)
            G = generate_nQudit_X(d, s) @ generate_nQudit_Z(d, t)
            C[(str(s), str(t))] = np.around(Tr(dag(X) @ G), 10)

    return C
