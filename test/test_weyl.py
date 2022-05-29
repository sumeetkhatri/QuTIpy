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

from qutipy.weyl import (
    discrete_Weyl,
    discrete_Weyl_X,
    discrete_Weyl_Z,
    generate_nQudit_X,
    generate_nQudit_Z,
    nQudit_cov_matrix,
    nQudit_quadratures,
    nQudit_Weyl_coeff,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_discrete_Weyl_X():
    assert np.all(
        discrete_Weyl_X(4)
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    assert np.all(
        discrete_Weyl_X(3)
        == np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )


def test_discrete_Weyl_Z():
    assert np.all(
        np.round(discrete_Weyl_Z(2), 5) == np.array([[1.0, 0.0], [0.0, -1.0]])
    )


def test_discrete_Weyl():
    # NOTE: Are discrete_Weyl_X and discrete_Weyl same ?
    assert np.all(
        np.round(discrete_Weyl(2, 1, 4), 5)
        == np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
    )


def test_nQudit_cov_matrix():
    assert np.all(
        np.round(nQudit_cov_matrix(X, 2, 2), 5)
        == np.array(
            [
                [34.0, -12.0, 34.0, -10.0],
                [12.0, 34.0, -20.0, 0.0],
                [34.0, -20.0, 34.0, -6.0],
                [-10.0, 0.0, 6.0, 34.0],
            ]
        )
    )


def test_nQudit_Weyl_coeff():
    coeff = nQudit_Weyl_coeff(H, 2, 1)

    assert np.all(
        [
            coeff[("[0]", "[0]")] == 0j,
            coeff[("[0]", "[1]")] == 1.4142135624 - 0j,
            coeff[("[1]", "[1]")] == 0j,
            coeff[("[1]", "[0]")] == 1.4142135624 + 0j,
        ]
    )


def test_nQudit_quadratures():
    quadratures = nQudit_quadratures(2, 2)
    assert np.all(
        quadratures[1]
        == np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        np.round(quadratures[2], 5)
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -0.0],
                [0.0, 0.0, -0.0, -1.0],
            ]
        )
    )
    assert np.all(
        np.round(quadratures[3], 5)
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    assert np.all(
        np.round(quadratures[4], 5)
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -0.0, 0.0, -1.0],
            ]
        )
    )
