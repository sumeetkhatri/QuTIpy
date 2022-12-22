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

from qutipy.general_functions import eye, partial_trace, trace_norm
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy


def norm_trace_dist(rho, sigma, sdp=False, dual=False, display=False):
    """
    Calculates the normalized trace distance (1/2)*||rho-sigma||_1 using an SDP,
    where rho and sigma are quantum states. More generally, they can be Hermitian
    operators.
    """

    if sdp:
        if not dual:

            dim = rho.shape[0]

            L1 = cvx.Variable((dim, dim), hermitian=True)
            L2 = cvx.Variable((dim, dim), hermitian=True)

            c = [L1 >> 0, L2 >> 0, eye(dim) - L1 >> 0, eye(dim) - L2 >> 0]

            obj = cvx.Maximize(cvx.real(cvx.trace((L1 - L2) @ (rho - sigma))))
            prob = cvx.Problem(obj, constraints=c)

            prob.solve(verbose=display, eps=1e-7)

            return (1 / 2) * prob.value

        elif dual:

            dim = rho.shape[0]

            Y1 = cvx.Variable((dim, dim), hermitian=True)
            Y2 = cvx.Variable((dim, dim), hermitian=True)

            c = [Y1 >> 0, Y2 >> 0, Y1 >> rho - sigma, Y2 >> -(rho - sigma)]

            obj = cvx.Minimize(cvx.real(cvx.trace(Y1 + Y2)))

            prob = cvx.Problem(obj, c)
            prob.solve(verbose=display, eps=1e-7)

            return (1 / 2) * prob.value
    else:
        return (1 / 2) * trace_norm(rho - sigma)


def norm_diamond_dist(J1, J2, dA, dB, dual=False, display=False):

    """
    Calculates the normalized diamond distance between two channels with
    Choi representations J1 and J2. For arbitrary superoperators, one can
    calculate this using the function diamond_norm as follows: (1/2)*diamond_norm(J1,J2,dA,dB).
    """

    if not dual:
        rho = cvx.Variable((dA, dA), hermitian=True)
        P = cvx.Variable((dA * dB, dA * dB), hermitian=True)

        c = [rho >> 0, P >> 0, P << cvx.kron(rho, eye(dB)), cvx.trace(rho) == 1]

        f = cvx.trace(P @ (J1 - J2))

        obj = cvx.Maximize(cvx.real(f))
        prob = cvx.Problem(obj, constraints=c)

        prob.solve(verbose=display)

        return prob.value

    else:

        mu = cvx.Variable()
        Z = cvx.Variable((dA, dB), hermitian=True)

        Z_A = numpy_to_cvxpy(partial_trace(cvxpy_to_numpy(Z), [2], [dA, dB]))

        c = [mu >= 0, Z >> 0, Z >> J1 - J2, mu * eye(dA) >> Z_A]

        obj = cvx.Minimize(mu)
        prob = cvx.Problem(obj, constraints=c)

        prob.solve(verbose=display)

        return prob.value
