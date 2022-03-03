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
from scipy.stats import unitary_group

from qutipy.general_functions import dag, eye, ket, syspermute, tensor


def CNOT_ij(i, j, n):
    """
    CNOT gate on qubits i and j, i being the control and j being the target.
    The total number of qubits is n.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)

    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, [i, j])

    perm_arrange = np.append(np.array([i, j]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)

    Sx = np.array([[0, 1], [1, 0]])
    CX = tensor(ket(2, 0) @ dag(ket(2, 0)), eye(2)) + tensor(
        ket(2, 1) @ dag(ket(2, 1)), Sx
    )

    out_temp = tensor(CX, [eye(2), n - 2])

    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def CZ_ij(i, j, n):
    """
    CZ gate on qubits i and j, i being the control and j being the target.
    The total number of qubits is n. (Note that for the CZ gate it does matter
    which qubit is the control and which qubit is the target.)
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)

    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, [i, j])

    perm_arrange = np.append(np.array([i, j]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)

    Sz = np.array([[1, 0], [0, -1]])
    CZ = tensor(ket(2, 0) @ dag(ket(2, 0)), eye(2)) + tensor(
        ket(2, 1) @ dag(ket(2, 1)), Sz
    )

    out_temp = tensor(CZ, [eye(2), n - 2])

    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def H_i(i, n):
    """
    Generates the matrix for the Hadamard gate applied to the ith qubit.
    n is the total number of qubits.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)
    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, i)
    perm_arrange = np.append(np.array([i]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    out_temp = tensor(H, [eye(2), n - 1])
    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def RandomUnitary(dim):
    """
    Generates a random unitary.
    """

    return unitary_group.rvs(dim)


def Rx_i(i, t, n):
    """
    Rotation about the X axis on qubit i by angle t. The total number of
    qubits is n.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)
    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, i)
    perm_arrange = np.append(np.array([i]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)
    Sx = np.array([[0, 1], [1, 0]])
    Rx = expm(-1j * t * Sx / 2)
    out_temp = tensor(Rx, [eye(2), n - 1])
    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def Rx(t):
    """
    Generates the unitary rotation matrix about the X axis on the Bloch sphere.
    """

    Sx = np.array([[0, 1], [1, 0]])

    return expm(-1j * t * Sx / 2.0)


def Ry_i(i, t, n):
    """
    Rotation about the Y axis on qubit i by angle t. The total number of
    qubits is n.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)
    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, i)
    perm_arrange = np.append(np.array([i]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)
    Sy = np.array([[0, -1j], [1j, 0]])
    Ry = expm(-1j * t * Sy / 2)
    out_temp = tensor(Ry, [eye(2), n - 1])
    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def Ry(t):
    """
    Generates the unitary rotation matrix about the Y axis on the Bloch sphere.
    """

    Sy = np.array([[0, -1j], [1j, 0]])

    return expm(-1j * t * Sy / 2.0)


def Rz_i(i, t, n):
    """
    Rotation about the Z axis on qubit i by angle t. The total number of
    qubits is n.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)
    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, i)
    perm_arrange = np.append(np.array([i]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)
    Sz = np.array([[1, 0], [0, -1]])
    Rz = expm(-1j * t * Sz / 2)
    out_temp = tensor(Rz, [eye(2), n - 1])
    out = syspermute(out_temp, perm_rearrange, dims)

    return out


def Rz(t):
    """
    Generates the unitary rotation matrix about the Z axis on the Bloch sphere.
    """

    Sz = np.array([[1, 0], [0, -1]])

    return expm(-1j * t * Sz / 2.0)


def S_i(i, n):
    """
    Generates the matrix for the S gate applied to the ith qubit.
    n is the total number of qubits. The S gate is defined as:

        S:=[[1 0],
            [0 1j]]

    It is one of the generators of the Clifford group.
    """

    dims = 2 * np.ones(n)
    dims = dims.astype(int)
    indices = np.linspace(1, n, n)
    indices_diff = np.setdiff1d(indices, i)
    perm_arrange = np.append(np.array([i]), indices_diff)
    perm_rearrange = np.zeros(n)

    for i in range(n):
        perm_rearrange[i] = np.argwhere(perm_arrange == i + 1)[0][0] + 1

    perm_rearrange = perm_rearrange.astype(int)
    S = np.array([[1, 0], [0, 1j]])
    out_temp = tensor(S, [eye(2), n - 1])
    out = syspermute(out_temp, perm_rearrange, dims)

    return out
