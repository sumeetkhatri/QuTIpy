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

from qutipy.general_functions import Tr, dag, eye, ket, tensor


def generate_nQubit_Pauli_X(indices):
    """
    Generates a tensor product of Pauli-X operators for n qubits. indices is
    a list of bits.
    """

    Id = eye(2)
    Sx = np.array([[0, 1], [1, 0]])

    out = 1

    for index in indices:
        if index == 0:
            out = tensor(out, Id)
        elif index == 1:
            out = tensor(out, Sx)
        else:
            return "Error: Indices must be bits, either 0 or 1!"

    return out


def generate_nQubit_Pauli_Z(indices):
    """
    Generates a tensor product of Pauli-Z operators for n qubits. indices is
    a list of bits.
    """

    Id = eye(2)
    Sz = np.array([[1, 0], [0, -1]])

    out = 1

    for index in indices:
        if index == 0:
            out = tensor(out, Id)
        elif index == 1:
            out = tensor(out, Sz)
        else:
            return "Error: Indices must be bits, either 0 or 1!"

    return out


def generate_nQubit_Pauli(indices):
    """
    Generates a tensor product of Pauli operators for n qubits. indices is a list
    of indices i specifying the Pauli operator for each tensor factor. i=0 is the identity, i=1 is sigma_x,
    i=2 is sigma_y, and i=3 is sigma_z.
    """

    Id = eye(2)
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    Sz = np.array([[1, 0], [0, -1]])

    out = 1

    for index in indices:
        if index == 0:
            out = tensor(out, Id)
        elif index == 1:
            out = tensor(out, Sx)
        elif index == 2:
            out = tensor(out, Sy)
        elif index == 3:
            out = tensor(out, Sz)

    return out


def nQubit_cov_matrix(X, n):
    """
    Using the n-qubit quadrature operators, we define the n-qubit "covariance matrix"
    as follows:

    V_{i,j}=Tr[X*S_i*S_j]
    """

    S = nQubit_quadratures(n)

    V = np.array(np.zeros((2 * n, 2 * n)), dtype=np.complex128)

    for i in range(2 * n):
        for j in range(2 * n):
            V[i, j] = Tr(X @ S[i + 1] @ S[j + 1])

    return V


def nQubit_mean_vector(X, n):
    """
    Using the n-qubit quadrature operators, we define the n-qubit "mean vector" as
    follows:

        r_i=Tr[X*S_i]
    """

    S = nQubit_quadratures(n)

    r = np.array(np.zeros((2 * n, 1)), dtype=np.complex128)

    for i in range(2 * n):
        r[i, 0] = Tr(X @ S[i + 1])

    return r


def nQubit_Pauli_coeff(X, n, return_dict=False):
    """
    Generates the coefficients of the matrix X in the n-qubit Pauli basis.
    The coefficients c_{alpha} are such that

    X=(1/2^n)\\sum_{alpha} c_alpha \\sigma_alpha

    The coefficients are returned in lexicographical ordering.
    """

    indices = list(itertools.product(*[range(0, 4)] * n))

    if return_dict:
        C = {}
    else:
        C = []

    for index in indices:
        sigma_i = generate_nQubit_Pauli(index)
        if return_dict:
            C[index] = Tr(dag(sigma_i) @ X)
        else:
            C.append(Tr(dag(sigma_i) @ X))

    return C


def nQubit_quadratures(n):
    """
    Returns the list of n-qubit "quadrature" operators, which are defined as
    (for two qubits)

        S[0]=Sx ⊗ Id
        S[1]=Sz ⊗ Id
        S[2]=Id ⊗ Sx
        S[3]=Id ⊗ Sz

    In general, for n qubits:

        S[0]=Sx ⊗ Id ⊗ ... ⊗ Id
        S[1]=Sz ⊗ Id ⊗ ... ⊗ Id
        S[2]=Id ⊗ Sx ⊗ ... ⊗ Id
        S[3]=Id ⊗ Sz ⊗ ... ⊗ Id
        .
        .
        .
        S[2n-2]=Id ⊗ Id ⊗ ... ⊗ Sx
        S[2n-1]=Id⊗ Id ⊗ ... ⊗ Sz
    """

    S = {}

    # Sx=np.array([[0,1],[1,0]])
    # Sz=np.array([[1,0],[0,-1]])

    count = 0

    for i in range(1, 2 * n + 1, 2):
        v = list(np.array(dag(ket(n, count)), dtype=int).flatten())
        S[i] = generate_nQubit_Pauli_X(v)
        S[i + 1] = generate_nQubit_Pauli_Z(v)
        count += 1

    return S


def Pauli_coeff_to_matrix(coeffs, n):
    """
    Takes the coefficients of a matrix in the n-qubit Pauli basis and outputs it
    as a matrix.

    coeffs should be specified as a one-dimensional list or array in standard
    lexicographical ordering.
    """

    all_indices = list(itertools.product(*[range(0, 4)] * n))

    out = 0 + 0j

    for i in range(len(all_indices)):
        out += (1.0 / 2.0**n) * coeffs[i] * generate_nQubit_Pauli(all_indices[i])

    return out
