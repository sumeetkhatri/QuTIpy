#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2023 Sumeet Khatri.
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

from qutipy.states import (
    GHZ,
    Bell,
    Werner_state,
    apply_discrete_Weyl_twirl,
    apply_isotropic_twirl,
    apply_Pauli_twirl,
    apply_Werner_twirl,
    graph_state,
    isotropic_state,
    max_ent,
    max_mix,
    random_density_matrix,
    random_state_vector,
    singlet_state,
)

dim = 3

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

Y = np.array(
    [
        [
            0.872 + 0.309j,
            0.963 + 0.12j,
            0.812 + 0.016j,
            0.463 + 0.984j,
            0.272 + 0.137j,
            0.319 + 0.428j,
            0.405 + 0.922j,
            0.744 + 0.459j,
            0.878 + 0.785j,
        ],
        [
            0.742 + 0.084j,
            0.393 + 0.841j,
            0.472 + 0.053j,
            0.646 + 0.117j,
            0.164 + 0.322j,
            0.264 + 0.576j,
            0.037 + 0.323j,
            0.3 + 0.062j,
            0.368 + 0.191j,
        ],
        [
            0.632 + 0.535j,
            0.315 + 0.067j,
            0.377 + 0.463j,
            0.026 + 0.227j,
            0.609 + 0.876j,
            0.23 + 0.114j,
            0.469 + 0.064j,
            0.724 + 0.081j,
            0.879 + 0.179j,
        ],
        [
            0.524 + 0.206j,
            0.232 + 0.403j,
            0.26 + 0.255j,
            0.759 + 0.295j,
            0.578 + 0.683j,
            0.721 + 0.871j,
            0.303 + 0.683j,
            0.485 + 0.501j,
            0.439 + 0.824j,
        ],
        [
            0.732 + 0.745j,
            0.932 + 0.403j,
            0.366 + 0.875j,
            0.918 + 0.708j,
            0.186 + 0.029j,
            0.101 + 0.822j,
            0.349 + 0.833j,
            0.137 + 0.936j,
            0.869 + 0.464j,
        ],
        [
            0.424 + 0.518j,
            0.545 + 0.851j,
            0.9 + 0.664j,
            0.926 + 0.579j,
            0.63 + 0.343j,
            0.979 + 0.17j,
            0.787 + 0.337j,
            0.421 + 0.568j,
            0.548 + 0.537j,
        ],
        [
            0.957 + 0.59j,
            0.816 + 0.426j,
            0.392 + 0.017j,
            0.381 + 0.174j,
            0.186 + 0.363j,
            0.406 + 0.656j,
            0.038 + 0.396j,
            0.819 + 0.022j,
            0.747 + 0.129j,
        ],
        [
            0.344 + 0.068j,
            0.201 + 0.527j,
            0.19 + 0.175j,
            0.993 + 0.118j,
            0.543 + 0.322j,
            0.985 + 0.35j,
            0.107 + 0.227j,
            0.234 + 0.959j,
            0.88 + 0.217j,
        ],
        [
            0.531 + 0.905j,
            0.182 + 0.633j,
            0.888 + 0.28j,
            0.98 + 0.181j,
            0.143 + 0.185j,
            0.583 + 0.501j,
            0.608 + 0.678j,
            0.653 + 0.042j,
            0.453 + 0.071j,
        ],
    ]
)


H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_max_ent():
    maxEnt = max_ent(dim)
    assert maxEnt.shape == (9, 9)
    i = maxEnt[0][0]
    assert np.all(
        maxEnt
        == np.array(
            [
                [i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i],
            ]
        )
    )


def test_random_density_matrix():
    rmd = random_density_matrix(dim)
    assert rmd.shape == (3, 3)


def test_max_mix():
    maxMix = max_mix(dim)
    assert np.all(
        maxMix == np.array([[1 / 3, 0.0, 0.0], [0.0, 1 / 3, 0.0], [0.0, 0.0, 1 / 3]])
    )


def test_random_density_matrix_MaxMix():
    rmd = random_density_matrix(dim)
    maxMix = max_mix(dim)
    assert maxMix.shape == rmd.shape
    assert "numpy.complex" in str(type(rmd[0][0]))


def test_Bell():
    d = 2
    z = 1
    x = 1
    assert np.all(
        np.round(Bell(d, z, x, as_matrix=True), 5)
        == np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.5 + 0.0j, -0.5 - 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.5 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        )
    )


def test_GHZ():
    ghz = GHZ(3, 2)
    initials = (
        [round(1 / 3, 5)] + [0] * 3 + [round(1 / 3, 5)] + [0] * 3 + [round(1 / 3, 5)]
    )
    assert np.all(
        np.round(ghz, 5)
        == np.array(
            [
                initials,
                [0] * 9,
                [0] * 9,
                [0] * 9,
                initials,
                [0] * 9,
                [0] * 9,
                [0] * 9,
                initials,
            ]
        )
    )


def test_graph_state():
    dim = 5
    random_graph = np.random.random((dim, dim))
    np.fill_diagonal(random_graph, 0)
    assert np.all(
        np.round(graph_state(random_graph, n=2), 5)
        == np.array([[0.5], [0.5], [0.5], [0.5]])
    )


def test_isotropic_state():
    assert np.all(
        np.round(isotropic_state(np.sqrt(1 / 2), 2, fidelity=True), 5)
        == np.array(
            [
                [0.40237, 0.0, 0.0, 0.30474],
                [0.0, 0.09763, 0.0, 0.0],
                [0.0, 0.0, 0.09763, 0.0],
                [0.30474, 0.0, 0.0, 0.40237],
            ]
        )
    )


def test_apply_isotropic_twirl():
    assert np.all(
        np.around(apply_isotropic_twirl(X, 2),4)
        == np.array([[11.3333,  0.    ,  0.    ,  5.6667],
       [ 0.    ,  5.6667,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  5.6667,  0.    ],
       [ 5.6667,  0.    ,  0.    , 11.3333]])
    )


def test_random_state_vector():
    assert random_state_vector([2, 2], rank=2).shape == (4, 1)


def test_singlet_state():
    assert np.all(
        np.round(singlet_state(3), 8)
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.16666667, 0.0, -0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.16666667, 0.0, 0.0, 0.0, -0.16666667, 0.0, 0.0],
                [0.0, -0.16666667, 0.0, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.16666667, 0.0, -0.16666667, 0.0],
                [0.0, 0.0, -0.16666667, 0.0, 0.0, 0.0, 0.16666667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.16666667, 0.0, 0.16666667, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def test_Werner_state():
    assert np.all(
        np.round(Werner_state(0.3, 2), 5)
        == np.array(
            [
                [0.23333, 0.0, 0.0, 0.0],
                [0.0, 0.26667, -0.03333, 0.0],
                [0.0, -0.03333, 0.26667, 0.0],
                [0.0, 0.0, 0.0, 0.23333],
            ]
        )
    )


def test_apply_Werner_twirl():
    assert np.all(
        np.round(apply_Werner_twirl(X, 2), 8)
        == np.array([[11.33333333,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  5.66666667,  5.66666667,  0.        ],
       [ 0.        ,  5.66666667,  5.66666667,  0.        ],
       [ 0.        ,  0.        ,  0.        , 11.33333333]])
    )


def test_apply_discrete_Weyl_twirl():
    assert np.all(
        np.round(apply_discrete_Weyl_twirl(Y, 3, 2), 3)
        == np.array([[ 4.533+1.227j,  0.   -0.j   ,  0.   -0.j   ,  0.   -0.j   ,
         0.   -0.j   ,  2.55 +5.682j,  0.   -0.j   ,  6.27 +4.545j,
         0.   -0.j   ],
       [-0.   +0.j   ,  4.23 +4.221j,  0.   -0.j   ,  4.377+2.106j,
         0.   -0.j   ,  0.   -0.j   ,  0.   -0.j   ,  0.   -0.j   ,
         2.934+3.216j],
       [-0.   +0.j   , -0.   +0.j   ,  4.11 +5.151j, -0.   +0.j   ,
         4.176+5.304j,  0.   -0.j   ,  5.058+2.451j,  0.   -0.j   ,
         0.   -0.j   ],
       [-0.   +0.j   ,  5.058+2.451j,  0.   -0.j   ,  4.11 +5.151j,
         0.   -0.j   ,  0.   -0.j   ,  0.   -0.j   ,  0.   +0.j   ,
         4.176+5.304j],
       [-0.   +0.j   , -0.   +0.j   ,  6.27 +4.545j, -0.   +0.j   ,
         4.533+1.227j,  0.   -0.j   ,  2.55 +5.682j,  0.   -0.j   ,
         0.   +0.j   ],
       [ 2.934+3.216j, -0.   +0.j   , -0.   +0.j   ,  0.   +0.j   ,
        -0.   +0.j   ,  4.23 +4.221j, -0.   +0.j   ,  4.377+2.106j,
         0.   -0.j   ],
       [-0.   +0.j   , -0.   +0.j   ,  4.377+2.106j, -0.   +0.j   ,
         2.934+3.216j,  0.   +0.j   ,  4.23 +4.221j,  0.   -0.j   ,
         0.   -0.j   ],
       [ 4.176+5.304j, -0.   +0.j   , -0.   +0.j   ,  0.   +0.j   ,
         0.   +0.j   ,  5.058+2.451j, -0.   +0.j   ,  4.11 +5.151j,
         0.   -0.j   ],
       [-0.   +0.j   ,  2.55 +5.682j, -0.   +0.j   ,  6.27 +4.545j,
         0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ,
         4.533+1.227j]])
    )


def test_apply_Pauli_twirl():
    assert np.all(
        np.round(apply_Pauli_twirl(X, 2), 8)
        == np.array(
            [
                [34.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 34.0 + 0.0j],
                [0.0 + 0.0j, 34.0 + 0.0j, 34.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 34.0 + 0.0j, 34.0 + 0.0j, 0.0 + 0.0j],
                [34.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 34.0 + 0.0j],
            ]
        )
    )
