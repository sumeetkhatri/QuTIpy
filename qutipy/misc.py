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
from cvxpy import bmat


def base_number_to_int(string, base):
    b = base
    string = string[::-1]
    return sum([string[k] * b**k for k in range(len(string))])


def cvxpy_to_numpy(cvx_obj):
    """
    Converts a cvxpy variable into a numpy array.
    """

    if cvx_obj.is_scalar():
        return np.array(cvx_obj)
    elif len(cvx_obj.shape) == 1:  # cvx_obj is a (column or row) vector
        return np.array(list(cvx_obj))
    else:  # cvx_obj is a matrix
        X = []
        for i in range(cvx_obj.shape[0]):
            x = [cvx_obj[i, j] for j in range(cvx_obj.shape[1])]
            X.append(x)
        X = np.array(X)
        return X


def numpy_to_cvxpy(np_obj):
    """
    Converts numpy array to cvxpy expression.
    """
    np_obj_list = np_obj.tolist()
    return bmat(np_obj_list)
