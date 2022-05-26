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

from qutipy.general_functions import Tr, dag, eye, ket


def coherence_vector_star_product(n1, n2, d):
    """
    Computes the star product between two coherence vectors corresponding to states, so that
    n1 and n2 (the coherence vectors) have length d^2-1 each.

    Definition taken from:

        "Characterization of the positivity of the density matrix in terms of
        the coherence vector representation"
        PHYSICAL REVIEW A 68, 062322 (2003)
    """

    # L=su_generators(d)
    g = su_structure_constants(d)[1]

    p = []

    for k in range(1, d**2):
        pk = 0
        for i in range(1, d**2):
            for j in range(1, d**2):
                pk += (d / 2) * n1[i - 1] * n2[j - 1] * g[(i, j, k)]
        p.append(pk)

    return np.array(p)


def state_from_coherence_vector(n, d, state=True):
    """
    Uses the supplied coherence vector n to generate the corresponding operator via

        X=(1/d)*(eye(d)+n*L),

    where L are the su(d) generators. n is a vector of length d^2.
    """

    L = su_generators(d)

    X = np.array(np.zeros((d, d)), dtype=np.complex128)

    for i in range(len(L)):
        X += (1 / d) * n[i] * L[i]

    return X


def su_generators(d):
    """
    Generates the basis (aka generators) of the Lie algebra su(d)
    corresponding to the Lie group SU(d). The basis has d^2-1 elements.

    All of the generators are traceless and Hermitian. After adding the
    identity matrix, they form an orthogonal basis for all dxd matrices.

    The orthogonality condition is

        Tr[S_i*S_j]=d*delta_{i,j}

    (This is a particular convention we use here; there are other conventions.)

    For d=2, we get the Pauli matrices.
    """

    S = []

    S.append(eye(d))

    for l in range(d):
        for k in range(l):
            S.append(
                np.sqrt(d / 2)
                * (ket(d, k) @ dag(ket(d, l)) + ket(d, l) @ dag(ket(d, k)))
            )
            S.append(
                np.sqrt(d / 2)
                * (-1j * ket(d, k) @ dag(ket(d, l)) + 1j * ket(d, l) @ dag(ket(d, k)))
            )

    for k in range(1, d):
        X = 0
        for j in range(k):
            X += ket(d, j) @ dag(ket(d, j))

        S.append(np.sqrt(d / (k * (k + 1))) * (X - k * ket(d, k) @ dag(ket(d, k))))

    return S


def su_structure_constants(d):
    """
    Generates the structure constants corresponding to the su(d)
    basis elements. They are defined as follows:

        f_{i,j,k}=(1/(1j*d^2))*Tr[S_k*[S_i,S_j]]

        g_{i,j,k}=(1/d^2)*Tr[S_k*{S_i,S_j}]

    """

    f = {}
    g = {}

    S = su_generators(d)

    for i in range(1, d**2):
        for j in range(1, d**2):
            for k in range(1, d**2):

                f[(i, j, k)] = (1 / (1j * d**2)) * Tr(
                    S[k] @ (S[i] @ S[j] - S[j] @ S[i])
                )

                g[(i, j, k)] = (1 / d**2) * Tr(S[k] @ (S[i] @ S[j] + S[j] @ S[i]))

    return f, g
