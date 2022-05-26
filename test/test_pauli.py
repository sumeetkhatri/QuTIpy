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

from qutipy.pauli import (
    Pauli_coeff_to_matrix,
    generate_nQubit_Pauli,
    generate_nQubit_Pauli_X,
    generate_nQubit_Pauli_Z,
    nQubit_cov_matrix,
    nQubit_mean_vector,
    nQubit_Pauli_coeff,
    nQubit_quadratures,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_generate_nQubit_Pauli_X():
    assert np.all(
        generate_nQubit_Pauli_X([1, 0])
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
        generate_nQubit_Pauli_X([1, 1])
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        generate_nQubit_Pauli_X([1, 0, 1])
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def test_generate_nQubit_Pauli_Z():
    assert np.all(
        generate_nQubit_Pauli_Z([1, 0])
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        )
    )
    assert np.all(
        generate_nQubit_Pauli_Z([1, 1])
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert np.all(
        generate_nQubit_Pauli_Z([1, 0, 1])
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -1.0, 0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, -0.0, -0.0, -0.0],
                [0.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -1.0, -0.0],
                [0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.0],
            ]
        )
    )


def test_generate_nQubit_Pauli():
    # NOTE: Are generate_nQubit_Pauli_X and generate_nQubit_Pauli same ?
    assert np.all(
        generate_nQubit_Pauli([1, 0])
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
        generate_nQubit_Pauli([1, 1])
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        generate_nQubit_Pauli([1, 0, 1])
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def test_nQubit_cov_matrix():
    assert np.all(
        nQubit_cov_matrix(X, 2)
        == np.array(
            [
                [34.0, -12.0, 34.0, -10.0],
                [12.0, 34.0, -20.0, 0.0],
                [34.0, -20.0, 34.0, -6.0],
                [-10.0, 0.0, 6.0, 34.0],
            ]
        )
    )


def test_nQubit_mean_vector():
    assert np.all(
        nQubit_mean_vector(X, 2) == np.array([34.0, -20.0, 34.0, -10.0]).reshape(4, 1)
    )


def test_nQubit_Pauli_coeff():
    assert np.all(
        nQubit_Pauli_coeff(X, 2)
        == np.array(
            [
                34.0,
                34.0,
                -6j,
                -10.0,
                34.0,
                34,
                -6j,
                -10,
                -12j,
                -12j,
                0j,
                0j,
                -20.0,
                -20,
                0j,
                0,
            ]
        )
    )


def test_nQubit_quadratures():

    quadratures = nQubit_quadratures(2)
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
        quadratures[2]
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
        quadratures[3]
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
        quadratures[4]
        == np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -0.0, 0.0, -1.0],
            ]
        )
    )


def test_Pauli_coeff_to_matrix():
    assert np.all(
        Pauli_coeff_to_matrix([1, 2, 3, 4], 1)
        == np.array([[2.5 + 0.0j, 1.0 - 1.5j], [1.0 + 1.5j, -1.5 + 0.0j]])
    )
