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

import cvxpy as cvx
import numpy as np

from qutipy.general_functions import eye, partial_trace, partial_transpose, trace_norm


def check_kext(rhoAB, dimA, dimB, k, display=False):
    """
    Checks if the bipartite state rhoAB is k-extendible.
    """

    all_sys = list(range(1, k + 2))
    dim = [dimA] + [dimB] * k

    t = cvx.Variable()
    R = cvx.Variable((dimA * dimB**k, dimA * dimB**k), hermitian=True)

    obj = cvx.Maximize(t)

    c = [R - t * eye(dimA * dimB**k) >> 0]

    for j in range(2, k + 2):

        sys = list(np.setdiff1d(all_sys, [1, j]))

        R_ABj = partial_trace(R, sys, dim)

        c.append(R_ABj == rhoAB)

    prob = cvx.Problem(obj, constraints=c)

    prob.solve(verbose=display)

    return prob.value, R.value


def log_negativity(rhoAB, dimA, dimB):
    """
    Returns the log-negativity of the bipartite state rhoAB, which is defined as

        E_N(rhoAB) = log_2 || rhoAB^{T_B} ||_1.

    This is a faithful entanglement measure if both A and B are qubits or if one of
    then is a qubit and the other a qutrit. Such states are entangled if and only if
    the log-negativity is positive.

    See "Computable measure of entanglement", Phys. Rev. A 65, 032314 (2002).
    """

    return np.log2(trace_norm(partial_transpose(rhoAB, [2], [dimA, dimB])))
