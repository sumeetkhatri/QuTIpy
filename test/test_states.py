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
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
        [28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        [37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
        [46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
        [55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
        [64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0],
        [73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0],
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
        apply_isotropic_twirl(X - 1, 2)
        == np.array(
            [
                [10, 0.0, 0.0, 5.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0, 10.0],
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
        np.round(apply_Werner_twirl(X - 1, 2), 8)
        == np.array(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ]
        )
    )


def test_apply_discrete_Weyl_twirl():
    assert np.all(
        apply_discrete_Weyl_twirl(Y, 3, 2).astype("int32")
        == np.array(
            [
                [368, 0, 0, 0, 0, 368, 0, 368, 0],
                [0, 368, 0, 368, 0, 0, 0, 0, 368],
                [0, 0, 368, 0, 368, 0, 368, 0, 0],
                [0, 368, 0, 368, 0, 0, 0, 0, 368],
                [0, 0, 368, 0, 368, 0, 368, 0, 0],
                [368, 0, 0, 0, 0, 368, 0, 368, 0],
                [0, 0, 368, 0, 368, 0, 368, 0, 0],
                [368, 0, 0, 0, 0, 368, 0, 368, 0],
                [0, 368, 0, 368, 0, 0, 0, 0, 368],
            ]
        )
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
