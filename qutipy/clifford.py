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

from qutipy.channels import apply_channel
from qutipy.gates import CNOT_ij, H_i, Rx_i, Ry_i, Rz_i, S_i
from qutipy.general_functions import (
    dag,
    eye,
    ket,
    trace_distance_pure_states,
    unitary_distance,
)


def Clifford_group_generators(n):
    """
    Outputs the generators of the n-qubit Clifford group.
    """

    G = []

    if n == 1:
        G = [H_i(1, 1), S_i(1, 1)]
    else:
        for i in range(1, n + 1):
            G.append(H_i(i, n))
            G.append(S_i(i, n))
            for j in range(1, n + 1):
                if i < j:
                    G.append(CNOT_ij(i, j, n))
                else:
                    continue

    return G


def Clifford_twirl_channel_one_qubit(K, rho, sys=1, dim=[2]):
    """
    Twirls the given channel with Kraus operators in K by the one-qubit
    Clifford group on the given subsystem (specified by sys).
    """

    n = int(np.log2(np.sum([d for d in dim])))

    C1 = eye(2**n)
    C2 = Rx_i(sys, np.pi, n)
    C3 = Rx_i(sys, np.pi / 2.0, n)
    C4 = Rx_i(sys, -np.pi / 2.0, n)
    C5 = Rz_i(sys, np.pi, n)
    C6 = Rx_i(sys, np.pi, n) @ Rz_i(sys, np.pi, n)
    C7 = Rx_i(sys, np.pi / 2.0, n) @ Rz_i(sys, np.pi, n)
    C6 = Rx_i(sys, np.pi, n) @ Rz_i(sys, np.pi, n)
    C8 = Rx_i(sys, -np.pi / 2.0, n) @ Rz_i(sys, np.pi, n)
    C9 = Rz_i(sys, np.pi / 2.0, n)
    C10 = Ry_i(sys, np.pi, n) @ Rz_i(sys, np.pi / 2.0, n)
    C11 = Ry_i(sys, -np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)
    C12 = Ry_i(sys, np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)
    C13 = Rz_i(sys, -np.pi / 2.0, n)
    C14 = Ry_i(sys, np.pi, n) @ Rz_i(sys, -np.pi / 2.0, n)
    C15 = Ry_i(sys, -np.pi / 2.0, n) @ Rz_i(sys, -np.pi / 2.0, n)
    C16 = Ry_i(sys, np.pi / 2.0, n) @ Rz_i(sys, -np.pi / 2.0, n)
    C17 = (
        Rz_i(sys, -np.pi / 2.0, n)
        @ Rx_i(sys, np.pi / 2.0, n)
        @ Rz_i(sys, np.pi / 2.0, n)
    )
    C18 = (
        Rz_i(sys, np.pi / 2.0, n)
        @ Rx_i(sys, np.pi / 2.0, n)
        @ Rz_i(sys, np.pi / 2.0, n)
    )
    C19 = Rz_i(sys, np.pi, n) @ Rx_i(sys, np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)
    C20 = Rx_i(sys, np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)
    C21 = (
        Rz_i(sys, np.pi / 2.0, n)
        @ Rx_i(sys, -np.pi / 2.0, n)
        @ Rz_i(sys, np.pi / 2.0, n)
    )
    C22 = (
        Rz_i(sys, -np.pi / 2.0, n)
        @ Rx_i(sys, -np.pi / 2.0, n)
        @ Rz_i(sys, np.pi / 2.0, n)
    )
    C23 = Rx_i(sys, -np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)
    C24 = Rx_i(sys, np.pi, n) @ Rx_i(sys, -np.pi / 2.0, n) @ Rz_i(sys, np.pi / 2.0, n)

    C = [
        C1,
        C2,
        C3,
        C4,
        C5,
        C6,
        C7,
        C8,
        C9,
        C10,
        C11,
        C12,
        C13,
        C14,
        C15,
        C16,
        C17,
        C18,
        C19,
        C20,
        C21,
        C22,
        C23,
        C24,
    ]

    rho_twirl = 0

    for i in range(len(C)):
        rho_twirl += (
            (1.0 / 24.0)
            * C[i]
            @ apply_channel(K, dag(C[i]) @ rho @ C[i], sys, dim)
            @ dag(C[i])
        )

    return rho_twirl, C


def generate_Clifford_group(n, display=False):
    """
    Generates the n-qubit Clifford group. The display variable is for testing
    purposes, and to see the progress through the code.

    Note that even for n=2, this code will take a long time to run! There are
    11520 elements of the two-qubit Clifford group!
    """

    G = Clifford_group_generators(n)

    def in_list(L, elem):

        # Last modified: 27 June 2019
        """
        Checks if the given unitary elem is in the list L.
        """

        x = 0
        for l in L:
            if (
                np.around(unitary_distance(l, elem), 10) == 0
            ):  # Check of the distance is zero (up to 10 decimal places)
                x = 1
                break

        return x

    C = [eye(2**n)]
    generated = False

    while not generated:

        tmp = []
        num_added = 0

        for c in C:
            for g in G:
                t1 = c @ g
                t2 = c @ dag(g)

                # t1 and t2 might be the same, in which case we add only one of the two to the list (if needed).
                # Also, t1 and t2 might already by in tmp (up to global phase),
                # so we need to check for that as well.
                if np.around(unitary_distance(t1, t2), 10) == 0:
                    if not in_list(C, t1) and not in_list(tmp, t1):
                        tmp.append(t1)
                        num_added += 1
                # if t1 and t2 are different, add both to the list (if needed).
                else:
                    if not in_list(C, t1) and not in_list(tmp, t1):
                        tmp.append(t1)
                        num_added += 1
                    if not in_list(C, t2) and not in_list(tmp, t2):
                        tmp.append(t2)
                        num_added += 1

        if num_added > 0:
            for t in tmp:
                C.append(t)
        else:
            generated = True

        if display:
            print(len(C))

    return C


def generate_state_2design(C, n, display=False):
    """
    Takes the n-qubit Clifford gates provided in C and returns a
    corresponding state 2-design. This uses the fact that the Clifford
    gates (for any n) form a unitary 2-design, and that any unitary
    t-design can be used to construct a state t-design.
    """

    def in_list(L, elem):
        """
        Checks if the given pure state elem is in the list L.
        """

        x = 0

        for l in L:
            if np.around(trace_distance_pure_states(l, elem), 10) == 0:
                x = 1
                break

        return x

    S = [ket(2**n, 0)]

    for c in C:
        s_test = c @ ket(2**n, 0)

        if not in_list(S, s_test):
            S.append(s_test)

        if display:
            print(len(S))

    return S
