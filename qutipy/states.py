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
from numpy.linalg import matrix_power, norm

from qutipy.gates import CZ_ij
from qutipy.general_functions import SWAP, Tr, dag, eye, ket, syspermute, tensor
from qutipy.weyl import discrete_Weyl_X, discrete_Weyl_Z


def MaxEnt_state(dim, normalized=True, density_matrix=True):
    """
    Generates the dim-dimensional maximally entangled state, which is defined as

    (1/sqrt(dim))*(|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If normalized=False, then the function returns the unnormalized maximally entangled
    vector.

    If density_matrix=True, then the function returns the state as a density matrix.
    """

    if normalized:
        Bell = (1.0 / np.sqrt(dim)) * np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if density_matrix:
            return Bell @ dag(Bell)
        else:
            return Bell
    else:
        Gamma = np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if density_matrix:
            return Gamma @ dag(Gamma)
        else:
            return Gamma


def Bell_state(d, z, x, density_matrix=False):
    """
    Generates a d-dimensional Bell state with 0 <= z,x <= d-1. These are defined as

    |Phi_{z,x}> = (Z(z)X(x) ⊗ I)|Phi^+>

    """

    Bell = MaxEnt_state(d, density_matrix=density_matrix)

    W_zx = matrix_power(discrete_Weyl_Z(d), z) @ matrix_power(discrete_Weyl_X(d), x)

    if density_matrix:
        out = tensor(W_zx, eye(d)) @ Bell @ tensor(dag(W_zx), eye(d))
        return out
    else:
        out = tensor(W_zx, eye(d)) @ Bell
        return out


def GHZ_state(dim, n, density_matrix=True):
    """
    Generates the n-party GHZ state in dim-dimensions for each party, which is defined as

        |GHZ_n> = (1/sqrt(dim))*(|0,0,...,0> + |1,1,...,1> + ... + |d-1,d-1,...,d-1>)

    If density_matrix=True, then the function returns the state as a density matrix.
    """

    GHZ = (1 / np.sqrt(dim)) * np.sum([ket(dim, [i] * n) for i in range(dim)], 0)

    if density_matrix:
        return GHZ @ dag(GHZ)
    else:
        return GHZ


def graph_state(A_G, n, density_matrix=False, return_CZ=False, alt=True):
    """
    Generates the graph state corresponding to the undirected graph G with n vertices.
    A_G denotes the adjacency matrix of G, which for an undirected graph is a binary
    symmetric matrix indicating which vertices are connected.

    See the following book chapter for a review:

        ``Cluster States'' in Compedium of Quantum Physics, pp. 96-105, by H. J. Briegel.

    """

    plus = (1 / np.sqrt(2)) * (ket(2, 0) + ket(2, 1))

    plus_n = tensor([plus, n])

    G = eye(2**n)

    for i in range(n):
        for j in range(i, n):
            if A_G[i, j] == 1:
                G = G @ CZ_ij(i + 1, j + 1, n)

    if density_matrix:
        plus_n = plus_n @ dag(plus_n)
        if return_CZ:
            return G @ plus_n @ dag(G), G
        else:
            return G @ plus_n @ dag(G)
    else:
        if return_CZ:
            return G @ plus_n, G
        else:
            return G @ plus_n


def isotropic_state(p, d, fidelity=False):
    """
    Generates the isotropic state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_Iso = p*|Bell><Bell|+(1-p)*eye(d^2)/d^2,

    where -1/(d^2-1)<=p<=1. Isotropic states are invariant under U ⊗ conj(U)
    for any unitary U, where conj(U) is the complex conjugate of U.

    If fidelity=True, then the function returns a different parameterization of
    the isotropic state in which the parameter p is the fidelity of the state
    with respect to the maximally entangled state.
    """

    Bell = MaxEnt_state(d)

    if fidelity:
        return p * Bell + ((1 - p) / (d**2 - 1)) * (eye(d**2) - Bell)
    else:
        return p * Bell + (1 - p) * eye(d**2) / d**2


def isotropic_twirl_state(X, d):
    """
    Applies the twirling channel

        X -> ∫ (U ⊗ conj(U))*X*(U ⊗ conj(U)).H dU

    to the input operator X acting on two d-dimensional systems.

    For d=2, this is equivalent to

        X -> (1/24)*sum_i (c_i ⊗ conj(c_i))*X*(c_i ⊗ conj(c_i)).H

    where the unitaries c_i form the one-qubit Clifford group (because the Clifford
    unitaries constitute a unitary 2-design).

    This channel takes any state rho and converts it to an isotropic state with
    the same fidelity to the maximally entangled state as rho.
    """

    G = MaxEnt_state(d, normalized=False, density_matrix=True)

    return (Tr(X) / (d**2 - 1) - Tr(G @ X) / (d * (d**2 - 1))) * eye(d**2) + (
        Tr(G @ X) / (d**2 - 1) - Tr(X) / (d * (d**2 - 1))
    ) @ G


def MaxMix_state(dim):
    """
    Generates the dim-dimensional maximally mixed state.
    """

    return eye(dim) / dim


def RandomDensityMatrix(dim, *args):
    """
    Generates a random density matrix.

    Optional argument is for the rank r of the state.

    Optional argument comp is for whether the state should have
    complex entries
    """

    args = np.array(args)

    if args.size == 0:
        r = dim
    else:
        r = args[0]

    gin = np.random.randn(dim, r) + 1j * np.random.randn(dim, r)
    rho = gin @ dag(gin)

    return rho / Tr(rho)


def RandomStateVector(dim, rank=None):
    """
    Generates a random pure state.

    For multipartite states, dim should be a list of dimensions for each
    subsystem. In this case, the rank variable is for the Schmidt rank. To specify
    the Schmidt rank, there has to be a bipartition of the systems, so that dim
    has only two elements.
    """

    if rank is None:
        if isinstance(dim, list):
            dim = np.prod(dim)

        # Generate the real and imaginary parts of the components using numbers
        # sampled from the standard normal distribution (normal distribution with
        # mean zero and variance 1).
        psi = dag(
            np.array([np.random.randn(dim)]) + 1j * np.array([np.random.randn(dim)])
        )

        psi = psi / norm(psi)

        return psi
    else:
        dimA = dim[0]
        dimB = dim[1]

        if rank is None:
            rank = max([dimA, dimB])
        else:
            k = rank

        psi_k = MaxEnt_state(k, density_matrix=False, normalized=False)
        a = dag(
            np.array([np.random.rand(dimA * k)])
            + 1j * np.array([np.random.rand(dimA * k)])
        )
        b = dag(
            np.array([np.random.rand(dimB * k)])
            + 1j * np.array([np.random.rand(dimB * k)])
        )

        psi_init = syspermute(tensor(a, b), [1, 3, 2, 4], [k, dimA, k, dimB])

        psi = tensor(dag(psi_k), eye(dimA * dimB)) @ psi_init

        psi = psi / norm(psi)

        return psi


def singlet_state(d, perp=False):
    """
    Generates the singlet state acting on two d-dimensional systems, which is defined
    as

        (1/(d^2-d))(eye(d^2)-F),

    where F is the swap operator given by SWAP([1,2],[d,d]) (see below).

    If perp=True, then the function also returns the state orthogonal to the singlet
    state, given by

        (1/(d^2+d))(eye(d^2)+F).
    """

    F = SWAP([1, 2], [d, d])

    singlet = (1 / (d**2 - d)) * (eye(d**2) - F)

    if perp:
        singlet_perp = (1 / (d**2 + d)) * (eye(d**2) + F)
        return singlet, singlet_perp
    else:
        return singlet


def Werner_state(p, d, alt_param=False):
    """
    Generates the Werner state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_W=p*singlet+(1-p)*singlet_perp,

    where singlet is the state defined as (1/(d^2-d))*(eye(d^2)-SWAP) and
    singlet_perp is the state defined as (1/(d^2+d))*(eye(d^2)+SWAP),
    where SWAP is the swap operator between two d-dimensional systems. The parameter
    p is between 0 and 1.

    Werner states are invariant under U ⊗ U for every unitary U.

    If alt_param=True, then the function returns a different parameterization of
    the Werner state in which the parameter p is between -1 and 1, and

        rho_W=(1/(d^2-d*p))*(eye(d^2)-p*SWAP)

    """

    if alt_param:
        F = SWAP([1, 2], [d, d])
        return (1 / (d**2 - d * p)) * (eye(d**2) - p * F)
    else:
        singlet, singlet_perp = singlet_state(d, perp=True)
        return p * singlet + (1 - p) * singlet_perp


def Werner_twirl_state(X, d):
    """
    Applies the twirling channel

        X -> ∫ (U ⊗ U)*rho*(U ⊗ U).H dU

    to the input operator X acting on two d-dimensional systems.

    For d=2, this is equivalent to

        X -> (1/24)*sum_i (c_i ⊗ c_i)*X*(c_i ⊗ c_i).H

    where the unitaries c_i form the one-qubit Clifford group (because the Clifford
    unitaries constitute a unitary 2-design).

    This channel takes any state rho and converts it to a Werner state with
    the same fidelity to the singlet state as rho.
    """

    F = SWAP([1, 2], [d, d])

    return (Tr(X) / (d**2 - 1) - Tr(F @ X) / (d * (d**2 - 1))) * eye(d**2) + (
        Tr(F @ X) / (d**2 - 1) - Tr(X) / (d * (d**2 - 1))
    ) @ F


# QuTIpy States Utility

import cvxpy as cvx

from qutipy.general_functions import partial_trace, partial_transpose, trace_norm


def check_kext(rhoAB, dimA, dimB, k, display=False):
    """
    Checks if the bipartite state rhoAB is k-extendible.
    """

    all_sys = list(range(1, k + 2))
    dim = [dimA] + [dimB] * k

    t = cvx.Variable()
    R = cvx.Variable((dimA * dimB**k, dimA * dimB**k), hermitian=True)

    obj = cvx.Maximize(t)

    c = [R - t * eye(dimA * dimB**k) >> 0]

    for j in range(2, k + 2):

        sys = list(np.setdiff1d(all_sys, [1, j]))

        R_ABj = partial_trace(R, sys, dim)

        c.append(R_ABj == rhoAB)

    prob = cvx.Problem(obj, constraints=c)

    prob.solve(verbose=display)

    return prob.value, R.value


def log_negativity(rhoAB, dimA, dimB):
    """
    Returns the log-negativity of the bipartite state rhoAB, which is defined as

        E_N(rhoAB) = log_2 || rhoAB^{T_B} ||_1.

    This is a faithful entanglement measure if both A and B are qubits or if one of
    then is a qubit and the other a qutrit. Such states are entangled if and only if
    the log-negativity is positive.

    See "Computable measure of entanglement", Phys. Rev. A 65, 032314 (2002).
    """

    return np.log2(trace_norm(partial_transpose(rhoAB, [2], [dimA, dimB])))
