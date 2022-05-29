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

from qutipy.fidelities import (
    avg_fidelity,
    avg_fidelity_qubit,
    ent_fidelity,
    ent_fidelity_channel,
    fidelity,
)
from qutipy.states import MaxEnt_state

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_avg_fidelity():
    assert np.round(avg_fidelity([H], 2), 8) == 0.33333333


def test_avg_fidelity_qubit():
    assert np.round(avg_fidelity_qubit(H @ (X / 20)[2:, 2:]), 5) == 0.9325


def test_ent_fidelity():
    assert np.round(ent_fidelity(X, 2), 8) == 17.0


def test_ent_fidelity_channel():
    assert np.round(ent_fidelity_channel([MaxEnt_state(2)], 4), 8) == 0.0625


def test_fidelity():
    assert fidelity(H, X[:2, :2]) == 13.178570359601418
    assert fidelity(H, X[::2, ::2]) == 24.91835598486949
