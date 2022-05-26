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

from qutipy.general_functions import (
    SWAP,
    Tr,
    dag,
    eye,
    generate_all_kets,
    get_subblock,
    ket,
    partial_trace,
    partial_transpose,
    permute_tensor_factors,
    spectral_norm,
    syspermute,
    tensor,
    trace_distance_pure_states,
    trace_norm,
    unitary_distance,
)
from qutipy.states import MaxEnt_state, MaxMix_state, RandomDensityMatrix

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_bra_ket():
    assert np.all(ket(2, 0) == np.array([[1.0], [0.0]]))


def test_dag():
    assert np.all(
        dag(X)
        == np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
    )


def test_eye():
    assert np.all(
        eye(5)
        == np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )


def test_generate_all_kets():
    assert np.all(
        generate_all_kets([2, 3])
        == np.array(
            [
                ket([2, 3], [0, 0]),
                ket([2, 3], [0, 1]),
                ket([2, 3], [0, 2]),
                ket([2, 3], [1, 0]),
                ket([2, 3], [1, 1]),
                ket([2, 3], [1, 2]),
            ]
        )
    )


def test_get_subblock():
    assert np.all(
        get_subblock(X, [1], [(1, 0)], [2, 2]) == np.array([[9, 10], [13, 14]])
    )


def test_partial_trace():
    assert partial_trace(X, [2], [2]) == 34


def test_partial_transpose():
    dimA = 2
    assert np.all(
        partial_transpose(MaxEnt_state(dimA, normalized=False), [2], [dimA, dimA])
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_permute_tensor_factors():
    permuted_tensor = permute_tensor_factors([1, 2], [2, 2])
    assert permuted_tensor.shape == (4, 4)
    assert (
        permuted_tensor[0, 0]
        == permuted_tensor[1, 1]
        == permuted_tensor[2, 2]
        == permuted_tensor[3, 3]
        == 1 + np.imag(0)
    )


def test_syspermute():
    # NOTE: Dims is [2, 2] and not [4, 4] as the dims are for states and not
    # for matrix
    assert np.all(
        syspermute(X, [2, 1], [2, 2])
        == np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])
    )


def test_spectral_norm():
    assert round(spectral_norm(X), 5) == 38.62266


def test_swap():
    assert np.all(
        SWAP([1, 2], [2, 2])
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_tensor():
    assert np.all(
        np.round(tensor(1, 2, 3, X, 0.2 + np.imag(0.3)), 5)
        == np.array(
            [
                [1.2, 2.4, 3.6, 4.8],
                [6.0, 7.2, 8.4, 9.6],
                [10.8, 12.0, 13.2, 14.4],
                [15.6, 16.8, 18.0, 19.2],
            ]
        )
    )


def test_Tr():
    assert Tr(X) == 34


def test_trace_distance_pure_states():
    state_1 = eye(4)
    state_2 = X
    assert trace_distance_pure_states(state_1, state_2) == -33


def test_trace_norm():
    assert np.round(trace_norm(X), 5) == 40.69398


def test_unitary_distance():
    assert unitary_distance(X, X) == -373
    assert unitary_distance(X, eye(4)) == -7.5
