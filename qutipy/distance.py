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

from qutipy.general_functions import eye, trace_norm


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
