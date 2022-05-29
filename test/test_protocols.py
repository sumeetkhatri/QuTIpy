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

from qutipy.channels import Choi_representation
from qutipy.channels import amplitude_damping_channel as damping_channel
from qutipy.protocols import (
    apply_ent_swap_GHZ_chain_channel,
    apply_ent_swap_GHZ_channel,
    apply_graph_state_dist_channel,
    apply_teleportation_chain_channel,
    apply_teleportation_channel,
    channel_discrimination,
    entanglement_distillation,
    post_ent_swap_GHZ_chain_fidelity,
    post_ent_swap_GHZ_fidelity,
    post_graph_state_dist_fidelity,
    post_teleportation_chain_fidelity,
    post_teleportation_fidelity,
    state_discrimination,
)
from qutipy.states import RandomDensityMatrix

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))

amplitude_damping_channel = [
    np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.8 + 0.0j]]),
    np.array([[0.0 + 0.0j, 0.4 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]),
]
Ag = np.array(
    [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]
)
RDM = np.array(
    [
        (0.0464 + 0j),
        (0.01737 + 0.00693j),
        (0.00283 - 0.01294j),
        (-0.00885 - 0.00835j),
        (-0.00357 + 0.00626j),
        (-0.00096 - 0.00152j),
        (-0.00936 + 0.00734j),
        (0.01414 - 0.01536j),
        (0.00249 - 0.01728j),
        (-0.00901 - 0.00331j),
        (0.01721 + 8e-05j),
        (0.00654 + 0.01068j),
        (-0.01177 + 0.008j),
        (0.00642 + 0.01061j),
        (0.00033 - 0.00684j),
        (-0.01445 + 0.00339j),
        (0.01737 - 0.00693j),
        (0.06047 + 0j),
        (-0.00987 - 0.00236j),
        (0.01548 + 0.00552j),
        (-0.00084 + 0.01862j),
        (0.00294 + 0.00458j),
        (-0.01311 + 0.00561j),
        (0.02119 - 0.00256j),
        (0.01295 - 0.00721j),
        (-0.0059 - 0.00658j),
        (0.00326 + 0.00492j),
        (0.00393 + 0.01097j),
        (0.00082 + 0.01521j),
        (0.00301 - 0.00827j),
        (0.00287 + 0.0087j),
        (0.01028 + 0.01191j),
        (0.00283 + 0.01294j),
        (-0.00987 + 0.00236j),
        (0.08097 + 0j),
        (-0.01827 - 0.01078j),
        (-0.00543 + 0.00634j),
        (0.00359 - 0.01574j),
        (0.008 + 0.00175j),
        (-0.00604 + 0.00544j),
        (-0.02118 - 0.02159j),
        (0.00413 - 0.01293j),
        (0.00939 + 0.02369j),
        (0.00724 - 0.01712j),
        (-0.00587 - 0.01569j),
        (-0.01254 - 0.01057j),
        (0.00261 + 0.00102j),
        (0.00127 - 0.00916j),
        (-0.00885 + 0.00835j),
        (0.01548 - 0.00552j),
        (-0.01827 + 0.01078j),
        (0.07494 - 0j),
        (0.01475 - 0.03589j),
        (0.00065 + 0.01032j),
        (-0.00219 - 0.00564j),
        (0.029 + 0.01002j),
        (0.01244 + 0.00864j),
        (-0.00441 - 0.00557j),
        (-0.00567 - 0.00259j),
        (0.0017 - 0.00887j),
        (-0.01587 + 0.00614j),
        (-0.01259 - 0.00177j),
        (0.01116 + 0.00735j),
        (-0.01601 - 0.01882j),
        (-0.00357 - 0.00626j),
        (-0.00084 - 0.01862j),
        (-0.00543 - 0.00634j),
        (0.01475 + 0.03589j),
        (0.06412 + 0j),
        (0.00217 + 0.00052j),
        (0.00214 + 0.01201j),
        (-0.0117 + 0.01175j),
        (-0.0049 - 0.02132j),
        (-0.0015 + 0.01521j),
        (0.00474 - 0.01449j),
        (0.0277 - 0.00371j),
        (-0.01193 - 0.00749j),
        (-0.01068 - 0.02127j),
        (-0.00617 + 0.0051j),
        (0.01334 - 0.02225j),
        (-0.00096 + 0.00152j),
        (0.00294 - 0.00458j),
        (0.00359 + 0.01574j),
        (0.00065 - 0.01032j),
        (0.00217 - 0.00052j),
        (0.03743 + 0j),
        (-0.0034 + 0.0071j),
        (-0.00035 - 0.00369j),
        (0.01226 - 0.01383j),
        (0.0188 - 2e-05j),
        (-0.00468 + 0.00414j),
        (-0.00117 - 0.00334j),
        (0.00429 + 0.00444j),
        (0.00938 - 0.00214j),
        (0.00982 - 0.00813j),
        (0.00914 + 0.00652j),
        (-0.00936 - 0.00734j),
        (-0.01311 - 0.00561j),
        (0.008 - 0.00175j),
        (-0.00219 + 0.00564j),
        (0.00214 - 0.01201j),
        (-0.0034 - 0.0071j),
        (0.04149 + 0j),
        (-0.00305 + 0.00442j),
        (-0.01256 + 0.00355j),
        (-0.00043 - 0.01475j),
        (0.00041 + 0.00134j),
        (0.01827 - 0.01518j),
        (0.01205 - 0.01277j),
        (-0.0014 + 0.00181j),
        (-0.00567 - 0.01164j),
        (-0.00082 - 0.02605j),
        (0.01414 + 0.01536j),
        (0.02119 + 0.00256j),
        (-0.00604 - 0.00544j),
        (0.029 - 0.01002j),
        (-0.0117 - 0.01175j),
        (-0.00035 + 0.00369j),
        (-0.00305 - 0.00442j),
        (0.05456 + 0j),
        (0.00865 - 0.00152j),
        (0.00654 - 0.01941j),
        (-0.00783 + 0.00584j),
        (0.00051 + 0.00433j),
        (-0.00082 + 0.00869j),
        (0.00305 + 0.01595j),
        (0.02332 - 0.01078j),
        (-0.005 - 0.00972j),
        (0.00249 + 0.01728j),
        (0.01295 + 0.00721j),
        (-0.02118 + 0.02159j),
        (0.01244 - 0.00864j),
        (-0.0049 + 0.02132j),
        (0.01226 + 0.01383j),
        (-0.01256 - 0.00355j),
        (0.00865 + 0.00152j),
        (0.09636 + 0j),
        (-0.0088 + 0.00733j),
        (0.0018 - 0.02452j),
        (-0.02118 + 0.00432j),
        (-0.01188 + 0.00565j),
        (0.0071 + 0.00232j),
        (0.01992 + 0.01489j),
        (0.00379 - 0.00137j),
        (-0.00901 + 0.00331j),
        (-0.0059 + 0.00658j),
        (0.00413 + 0.01293j),
        (-0.00441 + 0.00557j),
        (-0.0015 - 0.01521j),
        (0.0188 + 2e-05j),
        (-0.00043 + 0.01475j),
        (0.00654 + 0.01941j),
        (-0.0088 - 0.00733j),
        (0.05607 - 0j),
        (-0.02298 + 0.00207j),
        (0.01216 - 0.00107j),
        (0.01707 + 0.00746j),
        (0.0038 + 0.00471j),
        (0.01548 + 0.01154j),
        (0.01222 + 0.00678j),
        (0.01721 - 8e-05j),
        (0.00326 - 0.00492j),
        (0.00939 - 0.02369j),
        (-0.00567 + 0.00259j),
        (0.00474 + 0.01449j),
        (-0.00468 - 0.00414j),
        (0.00041 - 0.00134j),
        (-0.00783 - 0.00584j),
        (0.0018 + 0.02452j),
        (-0.02298 - 0.00207j),
        (0.04959 - 0j),
        (0.00921 - 0.01025j),
        (-0.00602 - 0.01137j),
        (-0.001 - 0.00764j),
        (-0.0059 + 0.00881j),
        (-0.00902 - 0.00245j),
        (0.00654 - 0.01068j),
        (0.00393 - 0.01097j),
        (0.00724 + 0.01712j),
        (0.0017 + 0.00887j),
        (0.0277 + 0.00371j),
        (-0.00117 + 0.00334j),
        (0.01827 + 0.01518j),
        (0.00051 - 0.00433j),
        (-0.02118 - 0.00432j),
        (0.01216 + 0.00107j),
        (0.00921 + 0.01025j),
        (0.08802 + 0j),
        (0.00599 - 0.01316j),
        (0.03114 - 0.01506j),
        (-0.02769 - 0.00153j),
        (-0.01107 - 0.01074j),
        (-0.01177 - 0.008j),
        (0.00082 - 0.01521j),
        (-0.00587 + 0.01569j),
        (-0.01587 - 0.00614j),
        (-0.01193 + 0.00749j),
        (0.00429 - 0.00444j),
        (0.01205 + 0.01277j),
        (-0.00082 - 0.00869j),
        (-0.01188 - 0.00565j),
        (0.01707 - 0.00746j),
        (-0.00602 + 0.01137j),
        (0.00599 + 0.01316j),
        (0.05709 + 0j),
        (-0.00296 - 0.00489j),
        (0.00363 - 0.01044j),
        (0.02113 - 0.00319j),
        (0.00642 - 0.01061j),
        (0.00301 + 0.00827j),
        (-0.01254 + 0.01057j),
        (-0.01259 + 0.00177j),
        (-0.01068 + 0.02127j),
        (0.00938 + 0.00214j),
        (-0.0014 - 0.00181j),
        (0.00305 - 0.01595j),
        (0.0071 - 0.00232j),
        (0.0038 - 0.00471j),
        (-0.001 + 0.00764j),
        (0.03114 + 0.01506j),
        (-0.00296 + 0.00489j),
        (0.05944 + 0j),
        (-0.01101 - 0.01837j),
        (-0.01134 + 0.01975j),
        (0.00033 + 0.00684j),
        (0.00287 - 0.0087j),
        (0.00261 - 0.00102j),
        (0.01116 - 0.00735j),
        (-0.00617 - 0.0051j),
        (0.00982 + 0.00813j),
        (-0.00567 + 0.01164j),
        (0.02332 + 0.01078j),
        (0.01992 - 0.01489j),
        (0.01548 - 0.01154j),
        (-0.0059 - 0.00881j),
        (-0.02769 + 0.00153j),
        (0.00363 + 0.01044j),
        (-0.01101 + 0.01837j),
        (0.06515 + 0j),
        (0.01132 + 0.0011j),
        (-0.01445 - 0.00339j),
        (0.01028 - 0.01191j),
        (0.00127 + 0.00916j),
        (-0.01601 + 0.01882j),
        (0.01334 + 0.02225j),
        (0.00914 - 0.00652j),
        (-0.00082 + 0.02605j),
        (-0.005 + 0.00972j),
        (0.00379 + 0.00137j),
        (0.01222 - 0.00678j),
        (-0.00902 + 0.00245j),
        (-0.01107 + 0.01074j),
        (0.02113 + 0.00319j),
        (-0.01134 - 0.01975j),
        (0.01132 - 0.0011j),
        (0.06791 + 0j),
    ]
).reshape((16, 16))


def avg_on(func, sample=1000):
    return sum([func() for _ in range(sample)]) / sample


def test_state_discrimination():
    assert np.round(state_discrimination(X, X.T, 0.121), 8) == -14.92301838


def test_apply_teleportation_chain_channel():
    X_64 = np.array([X] * 4).reshape((8, 8))
    assert np.all(
        np.round(apply_teleportation_chain_channel(X_64, 1, dA=1, dR=1, dB=2), 8)
        == np.array([[34.0 + 0.0j, 34.0 - 0.0j], [34.0 - 0.0j, 34.0 + 0.0j]])
    )


def test_post_graph_state_dist_fidelity():
    fidility = np.round(
        avg_on(
            lambda: post_graph_state_dist_fidelity(
                np.array([[1, 0], [0, 1]]), 2, np.round(RandomDensityMatrix(16), 6)
            ),
            1000,
        ),
        3,
    )
    assert 0.245 < fidility < 0.255


def test_post_teleportation_fidelity():
    avg_fidility = np.round(
        avg_on(lambda: post_teleportation_fidelity(RandomDensityMatrix(16)), 1000), 3
    )
    assert 0.244 < avg_fidility <= 0.256


def test_post_ent_swap_GHZ_chain_fidelity():
    avg_fidility = np.round(
        avg_on(
            lambda: post_ent_swap_GHZ_chain_fidelity(RandomDensityMatrix(64), 2), 1000
        ),
        3,
    )
    assert 0.058 < avg_fidility <= 0.065


def test_apply_ent_swap_GHZ_channel():
    assert apply_ent_swap_GHZ_channel(np.arange(256).reshape((16, 16)) / 256).shape == (
        8,
        8,
    )
    assert np.all(
        np.round(apply_ent_swap_GHZ_channel(np.arange(256).reshape((16, 16)) / 256), 2)
        == np.array(
            [
                [0.2, 0.2, 0.23, 0.23, 0.26, 0.26, 0.29, 0.29],
                [0.2, 0.2, 0.23, 0.23, 0.26, 0.26, 0.29, 0.29],
                [0.7, 0.7, 0.73, 0.73, 0.76, 0.76, 0.79, 0.79],
                [0.7, 0.7, 0.73, 0.73, 0.76, 0.76, 0.79, 0.79],
                [1.2, 1.2, 1.23, 1.23, 1.26, 1.26, 1.29, 1.29],
                [1.2, 1.2, 1.23, 1.23, 1.26, 1.26, 1.29, 1.29],
                [1.7, 1.7, 1.73, 1.73, 1.76, 1.76, 1.79, 1.79],
                [1.7, 1.7, 1.73, 1.73, 1.76, 1.76, 1.79, 1.79],
            ]
        )
    )


def test_apply_ent_swap_GHZ_chain_channel():
    X = np.round(
        avg_on(
            lambda: apply_ent_swap_GHZ_chain_channel(RandomDensityMatrix(64), 2), 1000
        ),
        3,
    )
    assert np.all(0.059 < X.diagonal()) and np.all(X.diagonal() < 0.065)


def test_channel_discrimination():
    J1 = Choi_representation(damping_channel(0.22), 2)
    J2 = Choi_representation(damping_channel(0.35), 2)
    assert (
        0.1993
        < np.round(avg_on(lambda: channel_discrimination(J1, J2, 2, 2, 0.2), 100), 5)
        < 0.1997
    )


def test_post_ent_swap_GHZ_fidelity():
    X = np.round(
        avg_on(lambda: post_ent_swap_GHZ_fidelity(RandomDensityMatrix(16)), 1000), 3
    )
    assert 0.120 < X < 0.130


def test_entanglement_distillation():
    assert np.all(
        np.round(entanglement_distillation(X, X @ Ag / 256), 8)
        == np.array(
            [
                [0.1328125, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def test_apply_graph_state_dist_channel():
    assert apply_graph_state_dist_channel(
        Ag, 4, RandomDensityMatrix(2 ** (2 * 4))
    ).shape == (256, 256)

    assert np.all(
        np.round(apply_graph_state_dist_channel(Ag[0:1, 0:1], 1, X), 3)
        == np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 4.0 + 0.0j],
                [0.0 + 0.0j, 6.0 + 0.0j, 7.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 10.0 + 0.0j, 11.0 + 0.0j, 0.0 + 0.0j],
                [13.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 16.0 + 0.0j],
            ]
        )
    )


def test_apply_teleportation_channel():
    teleportation_channel = np.round(
        avg_on(
            lambda: np.round(apply_teleportation_channel(RandomDensityMatrix(16)), 8),
            1000,
        ),
        3,
    )
    assert np.all(teleportation_channel.diagonal() <= teleportation_channel.max())
    assert np.all(teleportation_channel.min() <= teleportation_channel.diagonal())


def test_post_teleportation_chain_fidelity():
    assert np.round(post_teleportation_chain_fidelity(RDM, 1), 8) == 0.18003
