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

from qutipy.gates import (
    CNOT_ij,
    CZ_ij,
    H_i,
    RandomUnitary,
    Rx,
    Rx_i,
    Ry,
    Ry_i,
    Rz,
    Rz_i,
    S_i,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_CNOT_ij():
    assert np.all(
        CNOT_ij(1, 2, 2)
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
        CNOT_ij(2, 1, 2)
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    )


def test_CZ_ij():
    assert np.all(
        CZ_ij(1, 2, 2)
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        )
    )


def test_H_i():
    assert np.all(np.round(H_i(1, 1), 5) == np.round(H, 5))
    assert np.all(
        np.round(H_i(1, 2), 5)
        == np.array(
            [
                [0.70711, 0.0, 0.70711, 0.0],
                [0.0, 0.70711, 0.0, 0.70711],
                [0.70711, 0.0, -0.70711, -0.0],
                [0.0, 0.70711, -0.0, -0.70711],
            ]
        )
    )
    assert True


def test_RandomUnitary():
    assert RandomUnitary(2).shape == (2, 2)
    assert RandomUnitary(5).shape == (5, 5)


def test_Rx_i():
    assert np.all(
        np.round(Rx_i(1, 2, 2), 5)
        == np.array(
            [
                [0.5403, 0.0, 0.0 - 0.84147j, 0.0],
                [0.0, 0.5403, 0.0, 0.0 - 0.84147j],
                [0.0 - 0.84147j, 0.0, 0.5403, 0.0],
                [0.0, 0.0 - 0.84147j, 0.0, 0.5403],
            ]
        )
    )


def test_Rx():
    assert Rx(2).shape == (2, 2)
    assert np.all(
        np.round(Rx(2), 5) == np.array([[0.5403, -0.84147j], [-0.84147j, 0.5403]])
    )


def test_Ry_i():
    assert np.all(
        np.round(Ry_i(1, 2, 2), 5)
        == np.array(
            [
                [0.5403, 0.0, -0.84147, 0.0],
                [0.0, 0.5403, 0.0, -0.84147],
                [0.84147, 0.0, 0.5403, 0.0],
                [0.0, 0.84147, 0.0, 0.5403],
            ]
        )
    )


def test_Ry():
    assert Ry(2).shape == (2, 2)
    assert np.all(
        np.round(Ry(2), 5) == np.array([[0.5403, -0.84147], [0.84147, 0.5403]])
    )


def test_Rz_i():
    assert np.all(
        np.round(Rz_i(1, 2, 2), 5)
        == np.array(
            [
                [0.5403 - 0.84147j, 0.0, 0.0, 0.0],
                [0.0, 0.5403 - 0.84147j, 0.0, 0.0],
                [0.0, 0.0, 0.5403 + 0.84147j, 0.0],
                [0.0, 0.0, 0.0, 0.5403 + 0.84147j],
            ]
        )
    )


def test_Rz():
    assert Rz(2).shape == (2, 2)
    assert np.all(
        np.round(Rz(2), 5)
        == np.array([[0.5403 - 0.84147j, 0.0], [0.0, 0.5403 + 0.84147j]])
    )


def test_S_i():
    assert np.all(S_i(1, 1) == np.array([[1.0, 0.0], [0.0, 1j]]))
