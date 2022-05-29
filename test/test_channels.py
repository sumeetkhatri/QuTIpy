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

from qutipy.channels import (
    BB84_channel,
    Choi_representation,
    Choi_to_Natural,
    Kraus_representation,
    Natural_representation,
    Pauli_channel,
    Pauli_channel_coeffs,
    Pauli_channel_nQubit,
    amplitude_damping_channel,
    apply_channel,
    bit_flip_channel,
    channel_scalar_multiply,
    completely_dephasing_channel,
    compose_channels,
    dephasing_channel,
    depolarizing_channel,
    depolarizing_channel_n_uses,
    depolarizing_channel_nQubits,
    diamond_norm,
    generalized_amplitude_damping_channel,
    generate_channel_isometry,
    n_channel_uses,
    phase_damping_channel,
    tensor_channels,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_Choi_to_Natural():
    assert np.all(
        Choi_to_Natural(X, 2, 2)
        == np.array([[1, 3, 9, 11], [2, 4, 10, 12], [5, 7, 13, 15], [6, 8, 14, 16]])
    )


def test_bit_flip_channel():
    channel = bit_flip_channel(0.2)
    assert len(channel) == 3

    assert len(channel[0]) == 4
    assert (
        channel[0][0].shape
        == channel[0][1].shape
        == channel[0][2].shape
        == channel[0][3].shape
    )

    assert channel[1].shape == (8, 2)

    assert channel[2].shape == (8, 8)


def test_completely_dephasing_channel():
    # assert completely_dephasing_channel
    channel = completely_dephasing_channel(2)
    assert len(channel) == 4
    assert np.all(
        np.round(channel[0], 8) == np.array([[0.70710678, 0.0], [0.0, 0.70710678]])
    )
    assert np.all(channel[1] == np.array([0, 0, 0, 0]).reshape((2, 2)))
    assert np.all(channel[2] == np.array([0, 0, 0, 0]).reshape((2, 2)))

    assert np.all(
        np.round(channel[3], 8) == np.array([[0.70710678, 0.0], [0.0, -0.70710678]])
    )


def test_Kraus_representation():
    K = Kraus_representation(H, 2, 1)
    assert np.all(
        np.round(K[0], 8) == np.array([[0.92387953 + 0.0j, 0.38268343 + 0.0j]])
    )
    assert np.round(K[0], 8).shape == (1, 2)


def test_phase_damping_channel():
    channel = phase_damping_channel(0.2)
    assert np.all(np.round(channel[0], 8) == np.array([[1.0, 0.0], [0.0, 0.4472136]]))
    assert np.all(np.round(channel[1], 8) == np.array([[0.0, 0.0], [0.0, 0.89442719]]))


def test_generate_channel_isometry():
    x = generate_channel_isometry([H], 2, 2)
    assert np.all(x[0] == H) and np.all(x[1] == H)


def test_Pauli_channel_nQubit():
    p = X[:2, :2].flatten() + H.flatten()
    channel = Pauli_channel_nQubit(1, p)
    assert len(channel) == 3
    assert len(channel[0]) == 4
    assert channel[1].shape == (8, 2)
    assert channel[2].shape == (8, 8)


def test_apply_channel():
    assert apply_channel(X, X.T) == 53584


def test_amplitude_damping_channel():
    channel = amplitude_damping_channel(0.2)
    assert np.all(np.round(channel[0], 5) == np.array([[1.0, 0.0], [0.0, 0.89443]]))
    assert np.all(np.round(channel[1], 5) == np.array([[0.0, 0.44721], [0.0, 0.0]]))


def test_Natural_representation():
    assert np.all(
        np.round(Natural_representation([H]), 8)
        == np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5, 0.5],
            ]
        )
    )


def test_BB84_channel():
    # assert BB84_channel
    E = BB84_channel(0.2)
    assert np.all(
        [
            np.round(E[0][0], 8) == np.array([0.8, 0.0, 0.0, 0.8]).reshape((2, 2)),
            np.round(E[0][1], 8) == np.array([0.0, 0.4, 0.4, 0.0]).reshape((2, 2)),
            np.round(E[0][2], 8)
            == np.array([0.0 + 0.0j, 0.0 - 0.2j, 0.0 + 0.2j, 0.0 + 0.0j]).reshape(
                (2, 2)
            ),
            np.round(E[0][3], 8) == np.array([0.4, 0.0, 0.0, -0.4]).reshape((2, 2)),
        ]
    )
    assert np.all(
        np.round(E[1], 8)
        == np.array(
            [
                [0.8 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.4 + 0.0j],
                [0.0 + 0.0j, 0.0 - 0.2j],
                [0.4 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.8 + 0.0j],
                [0.4 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.2j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.4 + 0.0j],
            ]
        )
    )
    assert np.all(
        [
            np.round(E[2][0], 8)[0] == 0.8,
            np.round(E[2][0], 8)[4] == 0.0,
            np.round(E[2][1], 8)[0] == 0.0,
            np.round(E[2][1], 8)[4] == 0.4,
            np.round(E[2][2], 8)[0] == 0.0,
            np.round(E[2][2], 8)[4] == -0.2j,
            np.round(E[2][3], 8)[0] == 0.4,
            np.round(E[2][3], 8)[4] == 0.0,
            np.round(E[2][4], 8)[0] == 0.0,
            np.round(E[2][4], 8)[4] == 0.8,
            np.round(E[2][5], 8)[0] == 0.4,
            np.round(E[2][5], 8)[4] == 0.0,
            np.round(E[2][6], 8)[0] == 0.2j,
            np.round(E[2][6], 8)[4] == 0.0,
            np.round(E[2][7], 8)[0] == 0.0,
            np.round(E[2][7], 8)[4] == -0.4,
        ]
    )
    assert E[2].shape == (8, 8)


def test_Choi_representation():
    assert np.all(
        Choi_representation(X, 4)
        == np.array(
            [
                [276.0 + 0.0j, 304.0 + 0.0j, 332.0 + 0.0j, 360.0 + 0.0j],
                [304.0 + 0.0j, 336.0 + 0.0j, 368.0 + 0.0j, 400.0 + 0.0j],
                [332.0 + 0.0j, 368.0 + 0.0j, 404.0 + 0.0j, 440.0 + 0.0j],
                [360.0 + 0.0j, 400.0 + 0.0j, 440.0 + 0.0j, 480.0 + 0.0j],
            ]
        )
    )


def test_compose_channels():
    assert np.all(compose_channels([X]) == X)


def test_tensor_channels():
    assert np.all(compose_channels([X]) == X)


def test_depolarizing_channel_n_uses():
    assert np.all(
        np.round(depolarizing_channel_n_uses(0.2, 2, X, 2), 8)
        == np.array(
            [
                [3.0, 2.44444444, 2.68888889, 2.15111111],
                [4.64444444, 6.66666667, 3.76444444, 5.37777778],
                [7.08888889, 5.37777778, 10.33333333, 7.82222222],
                [6.99111111, 9.77777778, 10.02222222, 14.0],
            ]
        )
    )


def test_diamond_norm():
    assert (
        np.round(
            diamond_norm(Choi_representation(amplitude_damping_channel(0.2), 2), 2, 2),
            5,
        )
        == 1
    )


def test_depolarizing_channel_nQubits():
    channel = depolarizing_channel_nQubits(2, 0.2)

    assert np.all(
        np.round(channel[0], 8)
        == np.array(
            [
                0.89442719,
                0,
                0,
                0,
                0,
                0.89442719,
                0,
                0,
                0,
                0,
                0.89442719,
                0,
                0,
                0,
                0,
                0.89442719,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                0.11547005,
                0.11547005,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
                0,
                0,
                0,
                0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                -0.11547005,
                0,
                0,
                0.11547005,
                0,
                0,
                0,
            ]
        ).reshape((16, 4, 4))
    )


def test_dephasing_channel():
    channel = dephasing_channel(0.2)
    assert np.all(
        np.round(channel[0], 8)
        == np.array(
            [
                [[0.89442719 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.89442719 + 0.0j]],
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 0.0 - 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                [[0.4472136 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.4472136 + 0.0j]],
            ]
        )
    )

    assert np.all(
        np.round(channel[1], 8)
        == np.array(
            [
                [0.89442719 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j],
                [0.4472136 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.89442719 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.4472136 + 0.0j],
            ]
        )
    )

    assert channel[2].shape == (8, 8)


def test_generalized_amplitude_damping_channel():
    channel = generalized_amplitude_damping_channel(0.1, 0.2)

    assert np.all(
        np.round(channel, 8)
        == np.array(
            [
                [[0.89442719, 0.0], [0.0, 0.84852814]],
                [[0.0, 0.28284271], [0.0, 0.0]],
                [[0.42426407, 0.0], [0.0, 0.4472136]],
                [[0.0, 0.0], [0.14142136, 0.0]],
            ]
        )
    )


def test_n_channel_uses():
    channel = n_channel_uses(H, 2)
    assert np.all(
        np.round(channel, 8)
        == [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, -0.5, 0.5, -0.5]),
            np.array([0.5, 0.5, -0.5, -0.5]),
            np.array([0.5, -0.5, -0.5, 0.5]),
        ]
    )


def test_channel_scalar_multiply():
    assert np.all(
        np.round(channel_scalar_multiply(H, 0.2), 5)
        == np.array([[0.31623, 0.31623], [0.31623, -0.31623]])
    )


def test_Pauli_channel_coeffs():
    assert np.all(
        Pauli_channel_coeffs([X], 2)
        == [
            374.0,
            8.0,
            32.0,
            0.0,
            340.0,
            -8.0,
            32.0,
            0.0,
            238.0,
            8.0,
            -32.0,
            0.0,
            204.0,
            -8.0,
            -32.0,
            0.0,
        ]
    )


def test_Pauli_channel():
    E = Pauli_channel(0.16, 0.04, 0.16)
    assert np.all(
        [
            np.round(E[0][0], 8) == np.array([0.8, 0.0, 0.0, 0.8]).reshape((2, 2)),
            np.round(E[0][1], 8) == np.array([0.0, 0.4, 0.4, 0.0]).reshape((2, 2)),
            np.round(E[0][2], 8)
            == np.array([0.0 + 0.0j, 0.0 - 0.2j, 0.0 + 0.2j, 0.0 + 0.0j]).reshape(
                (2, 2)
            ),
            np.round(E[0][3], 8) == np.array([0.4, 0.0, 0.0, -0.4]).reshape((2, 2)),
        ]
    )
    assert np.all(
        np.round(E[1], 8)
        == np.array(
            [
                [0.8 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.4 + 0.0j],
                [0.0 + 0.0j, 0.0 - 0.2j],
                [0.4 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.8 + 0.0j],
                [0.4 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.2j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.4 + 0.0j],
            ]
        )
    )
    assert np.all(
        [
            np.round(E[2][0], 8)[0] == 0.8,
            np.round(E[2][0], 8)[4] == 0.0,
            np.round(E[2][1], 8)[0] == 0.0,
            np.round(E[2][1], 8)[4] == 0.4,
            np.round(E[2][2], 8)[0] == 0.0,
            np.round(E[2][2], 8)[4] == -0.2j,
            np.round(E[2][3], 8)[0] == 0.4,
            np.round(E[2][3], 8)[4] == 0.0,
            np.round(E[2][4], 8)[0] == 0.0,
            np.round(E[2][4], 8)[4] == 0.8,
            np.round(E[2][5], 8)[0] == 0.4,
            np.round(E[2][5], 8)[4] == 0.0,
            np.round(E[2][6], 8)[0] == 0.2j,
            np.round(E[2][6], 8)[4] == 0.0,
            np.round(E[2][7], 8)[0] == 0.0,
            np.round(E[2][7], 8)[4] == -0.4,
        ]
    )
    assert E[2].shape == (8, 8)


def test_depolarizing_channel():
    channel = depolarizing_channel(0.2)
    assert np.all(
        np.round(channel[0], 8)
        == np.array(
            [
                [[0.89442719 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.89442719 + 0.0j]],
                [[0.0 + 0.0j, 0.25819889 + 0.0j], [0.25819889 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 0.0 - 0.25819889j], [0.0 + 0.25819889j, 0.0 + 0.0j]],
                [[0.25819889 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.25819889 + 0.0j]],
            ]
        )
    )
