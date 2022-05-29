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

from qutipy.entropies import (
    Holevo_inf_channel,
    Holevo_inf_ensemble,
    Petz_Renyi_mut_inf_state,
    Petz_Renyi_rel_ent,
    bin_entropy,
    coherent_inf_channel,
    coherent_inf_state,
    entropy,
    hypo_testing_rel_ent,
    min_output_entropy,
    mutual_information,
    relative_entropy,
    relative_entropy_var,
    sandwiched_Renyi_mut_inf_state,
    sandwiched_Renyi_rel_ent,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))

amplitude_damping_channel = [
    np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.8 + 0.0j]]),
    np.array([[0.0 + 0.0j, 0.4 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]),
]


def test_relative_entropy_var():
    assert np.round(relative_entropy_var(H, X[2:, 2:]), 8) == -9.8180153


def test_mutual_information():
    assert np.round(mutual_information(X, 2, 2), 8) == -174.76652226


def test_bin_entropy():
    assert np.round(bin_entropy(0.121), 8) == 0.53222852


def test_sandwiched_Renyi_rel_ent():
    assert np.round(sandwiched_Renyi_rel_ent(H, X[2:, 2:], 0.121), 8) == -4.66949888


def test_coherent_inf_state():
    assert np.round(coherent_inf_state(X, 2, 2, s=0.121), 8) == 1.75732405


def test_sandwiched_Renyi_mut_inf_state():
    assert np.round(sandwiched_Renyi_mut_inf_state(X, 2, 2, 0.121, False), 2) == -11.09


def test_Petz_Renyi_mut_inf_state():
    assert np.round(Petz_Renyi_mut_inf_state(X, 2, 2, 0.121, False), 3) == -10.908


def test_Holevo_inf_ensemble():
    assert np.round(Holevo_inf_ensemble([0.121], [X]), 8) == 12.53503319


def test_Petz_Renyi_rel_ent():
    assert np.round(Petz_Renyi_rel_ent(H, X[2:, 2:], 0.121), 8) == -4.72051856


def test_coherent_inf_channel():
    assert (
        np.round(coherent_inf_channel(amplitude_damping_channel, 2, 2), 8) == 0.4568916
    )


def test_entropy():
    assert np.round(entropy(X), 8) == -184.97595895


def test_hypo_testing_rel_ent():
    assert np.round(hypo_testing_rel_ent(X, X.T, 0.121), 8) == 0.879


def test_relative_entropy():
    assert np.round(relative_entropy(X, X.T), 8) == -18.90321496


def test_Holevo_inf_channel():
    assert np.round(Holevo_inf_channel([H], 2), 8) == 1.0


def test_min_output_entropy():
    t = X @ X.transpose() / (X.mean() * 100)
    new_t = t @ t.transpose()
    new_t = new_t[:2, :2] + new_t[2:, 2:]
    new_t = (new_t @ amplitude_damping_channel @ H)[0]
    assert np.round(min_output_entropy([new_t], 2), 8) == -19.82683426
