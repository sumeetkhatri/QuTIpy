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
from scipy.linalg import sqrtm

from qutipy.channels import Choi_representation, apply_channel
from qutipy.general_functions import Tr, dag, ket, trace_norm
from qutipy.states import MaxEnt_state


def avg_fidelity_qubit(K):
    """
    K is the set of Kraus operators for the (qubit to qubit) channel whose
    average fidelity is to be found.
    """

    ket0 = ket(2, 0)
    ket1 = ket(2, 1)
    ket_plus = (1.0 / np.sqrt(2)) * (ket0 + ket1)
    ket_minus = (1.0 / np.sqrt(2)) * (ket0 - ket1)
    ket_plusi = (1.0 / np.sqrt(2)) * (ket0 + 1j * ket1)
    ket_minusi = (1.0 / np.sqrt(2)) * (ket0 - 1j * ket1)

    states = [ket0, ket1, ket_plus, ket_minus, ket_plusi, ket_minusi]

    F = 0

    for state in states:

        F += np.real(Tr((state @ dag(state)) * apply_channel(K, state @ dag(state))))

    return (1.0 / 6.0) * F


def avg_fidelity(K, dimA):
    """
    Calculates the average fidelity of a channel using its entanglement
    fidelity with respect to the maximally mixed state (see, e.g., Eq.
    (9.245) of Wilde's book.)

    K is the set of Kraus operators of the channel, and dimA is the dimension
    of the input space of the channel.
    """

    choi_state = (1.0 / dimA) * Choi_representation(K, dimA)

    return (dimA * ent_fidelity(choi_state, dimA) + 1) / (dimA + 1)


def ent_fidelity_channel(K, d):
    """
    Finds the entanglement fidelity of the channel given by the set K of
    Kraus operators. d is the dimension of the input space.
    """

    Bell = MaxEnt_state(d)

    K_choi = (1.0 / d) * Choi_representation(K, d)

    return np.real(Tr((Bell) @ K_choi))


def ent_fidelity(sigma, d):
    """
    Finds the fidelity between the state sigma and the Bell state.
    d is the dimension.
    """

    Bell = MaxEnt_state(d, density_matrix=True)

    return np.real(Tr(Bell @ sigma))


def fidelity(rho, sigma):
    """
    Returns the fidelity between the states rho and sigma.
    """

    return trace_norm(sqrtm(rho) @ sqrtm(sigma)) ** 2
