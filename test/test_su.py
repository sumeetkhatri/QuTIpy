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

from qutipy.su import (
    coherence_vector_star_product,
    state_from_coherence_vector,
    su_generators,
    su_structure_constants,
)

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_coherence_vector_star_product():
    # NOTE: Logic? Check with Dr. Khatri
    assert np.all(
        coherence_vector_star_product([1, 2, 3], [1, 2, 5], 2) == np.array([0, 0, 0])
    )


def test_state_from_coherence_vector():
    assert np.all(
        state_from_coherence_vector([33, 21, 1, 21], 2)
        == np.array([[27.0 + 0.0j, 10.5 - 0.5j], [10.5 + 0.5j, 6.0 + 0.0j]])
    )


def test_su_generators():
    special_unitaries = su_generators(2)
    assert len(special_unitaries) == 4
    assert np.all(special_unitaries[0] == np.array([1, 0, 0, 1]).reshape((2, 2)))
    assert np.all(special_unitaries[1] == np.array([0, 1, 1, 0]).reshape((2, 2)))
    assert np.all(special_unitaries[2] == np.array([0, -1j, 1j, 0]).reshape((2, 2)))
    assert np.all(special_unitaries[3] == np.array([1, 0, 0, -1]).reshape((2, 2)))


def test_su_structure_constants():
    f, g = su_structure_constants(2)
    assert len(f) == len(g) == 27
    assert f == {
        (1, 1, 1): 0j,
        (1, 1, 2): 0j,
        (1, 1, 3): 0j,
        (1, 2, 1): 0j,
        (1, 2, 2): 0j,
        (1, 2, 3): (1 + 0j),
        (1, 3, 1): 0j,
        (1, 3, 2): (-1 - 0j),
        (1, 3, 3): 0j,
        (2, 1, 1): 0j,
        (2, 1, 2): 0j,
        (2, 1, 3): (-1 - 0j),
        (2, 2, 1): 0j,
        (2, 2, 2): 0j,
        (2, 2, 3): 0j,
        (2, 3, 1): (1 + 0j),
        (2, 3, 2): 0j,
        (2, 3, 3): 0j,
        (3, 1, 1): 0j,
        (3, 1, 2): (1 + 0j),
        (3, 1, 3): 0j,
        (3, 2, 1): (-1 - 0j),
        (3, 2, 2): 0j,
        (3, 2, 3): 0j,
        (3, 3, 1): 0j,
        (3, 3, 2): 0j,
        (3, 3, 3): 0j,
    }
    assert g == {
        (1, 1, 1): 0.0,
        (1, 1, 2): 0j,
        (1, 1, 3): 0.0,
        (1, 2, 1): 0j,
        (1, 2, 2): 0j,
        (1, 2, 3): 0j,
        (1, 3, 1): 0.0,
        (1, 3, 2): 0j,
        (1, 3, 3): 0.0,
        (2, 1, 1): 0j,
        (2, 1, 2): 0j,
        (2, 1, 3): 0j,
        (2, 2, 1): 0j,
        (2, 2, 2): 0j,
        (2, 2, 3): 0j,
        (2, 3, 1): 0j,
        (2, 3, 2): 0j,
        (2, 3, 3): 0j,
        (3, 1, 1): 0.0,
        (3, 1, 2): 0j,
        (3, 1, 3): 0.0,
        (3, 2, 1): 0j,
        (3, 2, 2): 0j,
        (3, 2, 3): 0j,
        (3, 3, 1): 0.0,
        (3, 3, 2): 0j,
        (3, 3, 3): 0.0,
    }
