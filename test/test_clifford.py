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

from qutipy.clifford import (
    Clifford_group_generators,
    Clifford_twirl_channel_one_qubit,
    generate_Clifford_group,
    generate_state_2design,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))

amplitude_damping_channel = [
    np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.8 + 0.0j]]),
    np.array([[0.0 + 0.0j, 0.4 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]),
]


def test_Clifford_group_generators():
    clifford_group = Clifford_group_generators(2)
    assert np.all(
        np.round(clifford_group[0], 6)
        == np.array(
            [
                [0.707107, 0.0, 0.707107, 0.0],
                [0.0, 0.707107, 0.0, 0.707107],
                [0.707107, 0.0, -0.707107, -0.0],
                [0.0, 0.707107, -0.0, -0.707107],
            ]
        )
    )
    assert np.all(
        np.round(clifford_group[1], 6)
        == np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j],
            ]
        )
    )
    assert np.all(
        np.round(clifford_group[2], 6)
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    assert np.all(
        np.round(clifford_group[3], 6)
        == np.array(
            [
                [0.707107, 0.707107, 0.0, 0.0],
                [0.707107, -0.707107, 0.0, -0.0],
                [0.0, 0.0, 0.707107, 0.707107],
                [0.0, -0.0, 0.707107, -0.707107],
            ]
        )
    )
    assert np.all(
        np.round(clifford_group[4], 6)
        == np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j],
            ]
        )
    )


def test_Clifford_twirl_channel_one_qubit():
    K = amplitude_damping_channel
    rho = H
    c = Clifford_twirl_channel_one_qubit(K, rho)
    assert np.all(
        [
            np.round(c[0], 3)
            == np.array(
                [0.552 + 0.0j, 0.552 + 0.0j, 0.552 + 0.0j, -0.552 + 0.0j]
            ).reshape((2, 2)),
            np.round(c[1][0], 4) == np.array([1.0, 0.0, 0.0, 1.0]).reshape((2, 2)),
            np.round(c[1][1], 4) == np.array([0j, -1j, -1j, 0j]).reshape((2, 2)),
            np.round(c[1][2], 4)
            == np.array([(0.7071 + 0j), -0.7071j, -0.7071j, (0.7071 + 0j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][3], 4)
            == np.array([(0.7071 + 0j), 0.70710j, 0.70710j, (0.70710 + 0j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][4], 4) == np.array([-1j, 0j, 0j, 1j]).reshape((2, 2)),
            np.round(c[1][5], 4) == np.array([0j, 1 + 0j, -1 + 0j, 0j]).reshape((2, 2)),
            np.round(c[1][6], 4)
            == np.array([-0.7071j, 0.7071, -0.7071, 0.7071j]).reshape((2, 2)),
            np.round(c[1][7], 4)
            == np.array([-0.7071j, -0.7071, 0.7071, 0.7071j]).reshape((2, 2)),
            np.round(c[1][8], 4)
            == np.array([(0.7071 - 0.7071j), 0j, 0j, (0.7071 + 0.7071j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][9], 4)
            == np.array([0j, -0.7071 - 0.7071j, 0.7071 - 0.7071j, 0j]).reshape((2, 2)),
            np.round(c[1][10], 4)
            == np.array(
                [(0.5 - 0.5j), 0.5 + 0.5j, (-0.5 + 0.5j), (0.5 + 0.5j)]
            ).reshape((2, 2)),
            np.round(c[1][11], 4)
            == np.array(
                [(0.5 - 0.5j), (-0.5 - 0.5j), (0.5 - 0.5j), (0.5 + 0.5j)]
            ).reshape((2, 2)),
            np.round(c[1][12], 4)
            == np.array([(0.7071 + 0.7071j), 0j, 0j, (0.7071 - 0.7071j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][13], 4)
            == np.array([0j, (-0.7071 + 0.7071j), 0.7071 + 0.7071j, 0j]).reshape(
                (2, 2)
            ),
            np.round(c[1][14], 4)
            == np.array(
                [(0.5 + 0.5j), 0.5 - 0.5j, (-0.5 - 0.5j), (0.5 - 0.5j)]
            ).reshape((2, 2)),
            np.round(c[1][15], 4)
            == np.array(
                [(0.5 + 0.5j), (-0.5 + 0.5j), 0.5 + 0.5j, (0.5 - 0.5j)]
            ).reshape((2, 2)),
            np.round(c[1][16], 4)
            == np.array(
                [(0.7071 + 0j), 0.7071 + 0j, -0.7071 + 0j, (0.7071 + 0j)]
            ).reshape((2, 2)),
            np.round(c[1][17], 4)
            == np.array([-0.7071j, -0.7071j, -0.7071j, 0.7071j]).reshape((2, 2)),
            np.round(c[1][18], 4)
            == np.array(
                [(-0.5 - 0.5j), -0.5 - 0.5j, 0.5 - 0.5j, (-0.5 + 0.5j)]
            ).reshape((2, 2)),
            np.round(c[1][19], 4)
            == np.array([(0.5 - 0.5j), 0.5 - 0.5j, -0.5 - 0.5j, (0.5 + 0.5j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][20], 4)
            == np.array([-0.7071j, 0.7071j, 0.7071j, 0.7071j]).reshape((2, 2)),
            np.round(c[1][21], 4)
            == np.array([(0.7071 + 0j), -0.7071, 0.7071, (0.7071 + 0j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][22], 4)
            == np.array([(0.5 - 0.5j), -0.5 + 0.5j, 0.5 + 0.5j, (0.5 + 0.5j)]).reshape(
                (2, 2)
            ),
            np.round(c[1][23], 4)
            == np.array([0.5 - 0.5j, 0.5 - 0.5j, -0.5 - 0.5j, 0.5 + 0.5j]).reshape(
                (2, 2)
            ),
        ]
    )


def test_generate_Clifford_group():
    # FIXME: This function is very slow for group > 1
    group = generate_Clifford_group(1)
    assert np.all(
        np.round(group, 4)
        == np.array(
            [
                1,
                0,
                0,
                1,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                -0.7071 + 0.0j,
                1,
                0,
                0,
                +1.0j,
                1,
                0,
                0,
                -1.0j,
                0.7071 + 0.0j,
                +0.7071j,
                0.7071 + 0.0j,
                -0.7071j,
                0.7071 + 0.0j,
                -0.7071j,
                0.7071 + 0.0j,
                +0.7071j,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                +0.7071j,
                -0.7071j,
                1,
                0,
                0,
                -1,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                -0.7071j,
                +0.7071j,
                0.5 + 0.5j,
                0.5 - 0.5j,
                0.5 - 0.5j,
                0.5 + 0.5j,
                0.7071 + 0.0j,
                -0.7071 + 0.0j,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                0.5 - 0.5j,
                0.5 + 0.5j,
                0.5 + 0.5j,
                0.5 - 0.5j,
                0.7071 + 0.0j,
                -0.7071j,
                +0.7071j,
                -0.7071 + 0.0j,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                -0.7071 + 0.0j,
                0.7071 + 0.0j,
                0.7071 + 0.0j,
                +0.7071j,
                -0.7071j,
                -0.7071 + 0.0j,
                0.5 + 0.5j,
                -0.5 - 0.5j,
                0.5 - 0.5j,
                0.5 - 0.5j,
                -0,
                1,
                1,
                -0,
                0.5 - 0.5j,
                -0.5 + 0.5j,
                0.5 + 0.5j,
                0.5 + 0.5j,
                0.5 - 0.5j,
                0.5 + 0.5j,
                -0.5 + 0.5j,
                0.5 + 0.5j,
                0.7071 + 0.0j,
                -0.7071j,
                -0.7071 + 0.0j,
                -0.7071j,
                0,
                0.7071 + 0.7071j,
                0.7071 - 0.7071j,
                -0.0j,
                -0,
                -1.0j,
                1,
                0,
                0.5 - 0.5j,
                -0.5 + 0.5j,
                -0.5 + 0.5j,
                -0.5 + 0.5j,
                0,
                -0.7071 + 0.7071j,
                0.7071 - 0.7071j,
                0,
            ]
        ).reshape((24, 2, 2))
    )


def test_generate_state_2design():
    assert np.all(
        np.round(generate_state_2design([H], 1), 8)
        == np.array([[[1], [0]], [[0.70710678], [0.70710678]]])
    )
