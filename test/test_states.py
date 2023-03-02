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

Y = np.array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [10., 11., 12., 13., 14., 15., 16., 17., 18.],
       [19., 20., 21., 22., 23., 24., 25., 26., 27.],
       [28., 29., 30., 31., 32., 33., 34., 35., 36.],
       [37., 38., 39., 40., 41., 42., 43., 44., 45.],
       [46., 47., 48., 49., 50., 51., 52., 53., 54.],
       [55., 56., 57., 58., 59., 60., 61., 62., 63.],
       [64., 65., 66., 67., 68., 69., 70., 71., 72.],
       [73., 74., 75., 76., 77., 78., 79., 80., 81.]])


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
        apply_isotropic_twirl(X - 1, 2)
        == np.array(
            [
                [10 ,  0.0,  0.0,  5. ],
                [0. ,  5. ,  0.0,  0. ],
                [0. ,  0. ,  5.0,  0. ],
                [5. ,  0. ,  0. ,  10.],
            ]
        )
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
        np.round(apply_Werner_twirl(X-1, 2), 8)
        == np.array(
            [
                [10.,  0.,  0.,  0.],
                [ 0.,  5.,  5.,  0.],
                [ 0.,  5.,  5.,  0.],
                [ 0.,  0.,  0., 10.],
            ]
        )
    )


def test_apply_discrete_Weyl_twirl():
    assert np.all(
        apply_discrete_Weyl_twirl(Y, 3, 2).astype('int32')
        == np.array(
        [
           [369,   0,   0,   0,   0, 369,   0, 369,   0],
           [  0, 369,   0, 369,   0,   0,   0,   0, 369],
           [  0,   0, 369,   0, 369,   0, 369,   0,   0],
           [  0, 369,   0, 369,   0,   0,   0,   0, 369],
           [  0,   0, 369,   0, 369,   0, 369,   0,   0],
           [369,   0,   0,   0,   0, 369,   0, 369,   0],
           [  0,   0, 369,   0, 369,   0, 369,   0,   0],
           [369,   0,   0,   0,   0, 369,   0, 369,   0],
           [  0, 369,   0, 369,   0,   0,   0,   0, 369]
        ])
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
