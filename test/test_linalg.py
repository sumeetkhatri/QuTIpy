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

from qutipy.linalg import gram_schmidt, proj, rank

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_gram_schmidt():
    assert np.all(
        np.round(gram_schmidt([X], 4)[0], 8)
        == np.array(
            [
                [0.02585438, 0.05170877, 0.07756315, 0.10341754],
                [0.12927192, 0.15512631, 0.18098069, 0.20683508],
                [0.23268946, 0.25854384, 0.28439823, 0.31025261],
                [0.336107, 0.36196138, 0.38781577, 0.41367015],
            ]
        )
    )


def test_proj():
    assert np.all(
        np.round(proj(X[1], X[2]), 8)
        == np.array([7.98850575, 9.5862069, 11.18390805, 12.7816092])
    )


def test_rank():
    assert rank(X) == 2
