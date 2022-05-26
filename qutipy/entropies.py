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
from numpy.linalg import matrix_power, norm
from scipy.linalg import fractional_matrix_power, logm
from scipy.optimize import minimize

from qutipy.channels import apply_channel
from qutipy.general_functions import Tr, dag, eye, ket, partial_trace, tensor


def relative_entropy_var(rho, sigma):
    """
    Returns the relative entropy variance of rho and sigma, defined as

    V(rho||sigma)=Tr[rho*(log2(rho)-log2(sigma))^2]-D(rho||sigma)^2.
    """

    return (
        np.real(
            Tr(
                rho
                @ matrix_power((logm(rho)) / np.log(2) - (logm(sigma)) / np.log(2), 2)
            )
        )
        - relative_entropy(rho, sigma) ** 2
    )


def mutual_information(rhoAB, dimA, dimB):
    """
    Computes the mutual information of the bipartite state rhoAB, defined as

    I(A;B)_rho=D(rhoAB||rhoA⊗ rhoB)
    """

    rhoA = partial_trace(rhoAB, [2], [dimA, dimB])
    rhoB = partial_trace(rhoAB, [1], [dimA, dimB])

    return relative_entropy(rhoAB, tensor(rhoA, rhoB))


def bin_entropy(p):
    """
    Returns the binary entropy for 0<=p<=1.
    """

    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def sandwiched_Renyi_rel_ent(rho, sigma, alpha):
    """
    Computes the sandwiched Renyi relative entropy for either 0<=alpha<=1,
    or for alpha>=1 provided that supp(rho) is contained in supp(sigma).
    """

    sigma_a = np.array(fractional_matrix_power(sigma, (1.0 - alpha) / (2 * alpha)))

    Q = np.real(Tr(fractional_matrix_power(sigma_a @ rho @ sigma_a, alpha)))

    return (1.0 / (alpha - 1)) * np.log2(Q)


def coherent_inf_state(rho_AB, dimA, dimB, s=1):
    """
    Calculates the coherent information of the state rho_AB.

    If s=2, then calculates the reverse coherent information.
    """

    if s == 1:  # Calculate I_c(A>B)=H(B)-H(AB)
        rho_B = partial_trace(rho_AB, [1], [dimA, dimB])
        return entropy(rho_B) - entropy(rho_AB)
    else:  # Calculate I_c(B>A)=H(A)- H(AB) (AKA reverse coherent information)
        rho_A = partial_trace(rho_AB, [2], [dimA, dimB])
        return entropy(rho_A) - entropy(rho_AB)


def sandwiched_Renyi_mut_inf_state(rhoAB, dimA, dimB, alpha, opt=True):
    """
    Computes the sandwiched Renyi mutual information of the bipartite state
    rhoAB for 0<=alpha<=infty.

    TODO: Figure out how to do the computation with optimization over sigmaB.
    """

    rhoA = partial_trace(rhoAB, [2], [dimA, dimB])
    rhoB = partial_trace(rhoAB, [1], [dimA, dimB])

    if not opt:
        return sandwiched_Renyi_rel_ent(rhoAB, tensor(rhoA, rhoB), alpha)
    else:
        return None


def Petz_Renyi_mut_inf_state(rhoAB, dimA, dimB, alpha, opt=True):
    """
    Computes the Petz-Renyi mutual information of the bipartite state
    rhoAB for 0<=alpha<=1.

    TODO: Figure out how to do the computation with optimization over sigmaB.
    """

    rhoA = partial_trace(rhoAB, [2], [dimA, dimB])
    rhoB = partial_trace(rhoAB, [1], [dimA, dimB])

    if not opt:
        return Petz_Renyi_rel_ent(rhoAB, tensor(rhoA, rhoB), alpha)
    else:
        return None


def Holevo_inf_ensemble(p, S):
    """
    Computes the Holevo information of an ensemble.

    p is an array of probabilities, and S is an array of states.

    Based on MATLAB code written by Felix Lediztky.
    """

    dim = np.shape(S[0])[0]

    R = np.zeros((dim, dim))
    av = 0

    for i in range(len(p)):
        R = R + p[i] * S[i]
        av = av + p[i] * entropy(S[i])

    return entropy(R) - av


def Petz_Renyi_rel_ent(rho, sigma, alpha):
    """
    Computes the Petz-Renyi relative entropy of rho and sigma for 0<=alpha<=1.
    """

    rho_a = fractional_matrix_power(rho, alpha)
    sigma_a = fractional_matrix_power(sigma, 1 - alpha)

    Q = np.real(Tr(rho_a @ sigma_a))

    return (1.0 / (alpha - 1)) * np.log2(Q)


def coherent_inf_channel(K, dim_in, dim_out, s=1, display=True):
    """
    Calculates the coherent information of the channel specified by
    the Kraus operators in K.

    If s=2, then calculates the reverse coherent information of the channel.
    """

    def objfunc(x):
        Re = np.array(x[0 : dim_in**2])
        Re = np.reshape(Re, [*Re.shape, 1][:2])  # Build a matrix instead of a list

        Im = np.array(x[dim_in**2 :])
        Im = np.reshape(Im, [*Im.shape, 1][:2])  # Build a matrix instead of a list

        psi = np.array(Re.T + 1j * Im.T)
        psi = psi / norm(psi)

        psi_AA = psi * dag(psi)

        rho_AB = apply_channel(K, psi_AA, 2, dim=[dim_in, dim_in])

        return -coherent_inf_state(rho_AB, dim_in, dim_out, s)

    x_init = np.random.rand(2 * dim_in**2)

    opt = minimize(objfunc, x_init, options={"disp": display})

    return np.max([0, -opt.fun])


def entropy(rho):
    """
    Returns the quantum (von Neumann) entropy of the state rho.
    """

    return -np.real(Tr(rho @ logm(rho))) / np.log(2)


def hypo_testing_rel_ent(
    rho, sigma, eps, dual=False, log=False, display=False, return_all=False
):
    """
    Calculates the eps-hypothesis testing relative entropy of the two states
    rho and sigma.
    """

    if not dual:

        dim = rho.shape[0]

        L = cvx.Variable((dim, dim), hermitian=True)

        c = [L >> 0, eye(dim) - L >> 0]
        c += [cvx.real(cvx.trace(L @ rho)) >= 1 - eps]

        obj = cvx.Minimize(cvx.real(cvx.trace(L @ sigma)))
        prob = cvx.Problem(obj, constraints=c)

        prob.solve(verbose=display, eps=1e-7)

        if not log:
            if return_all:
                return prob.value, L.value
            else:
                return prob.value
        else:
            if return_all:
                return -np.log2(prob.value), L.value
            else:
                return -np.log2(prob.value)

    elif dual:

        dim = rho.shape[0]

        Z = cvx.Variable((dim, dim), hermitian=True)
        mu = cvx.Variable()

        c = [mu >= 0, mu * rho << sigma + Z, Z >> 0]

        obj = cvx.Maximize(cvx.real(mu * (1 - eps) - cvx.trace(Z)))

        prob = cvx.Problem(obj, c)
        prob.solve(verbose=display, eps=1e-7)

        if not log:
            if return_all:
                return prob.value, Z.value, mu.value
            else:
                return prob.value
        else:
            if return_all:
                return -np.log2(prob.value), Z.value, mu.value
            else:
                return -np.log2(prob.value)


def relative_entropy(rho, sigma):
    """
    Computes the standard (von Neumann) quantum relative entropy of rho
    and sigma, provided that supp(rho) is contained in supp(sigma).
    """

    return np.real(Tr(rho @ (logm(rho) - logm(sigma)))) / np.log(2)


def Holevo_inf_channel(K, dim, display=True):
    """
    Computes the Holevo information of a channel given by its set of
    Kraus operators K. dim is the dimension of the input space of the
    channel.

    Based on MATLAB code written by Felix Leditzky.
    """

    def objfunc(x):
        Re = np.array(x[0 : dim**3])
        Im = np.array(x[dim**3 :])

        psi = np.array([Re + 1j * Im]).T
        psi = psi / norm(psi)

        p = []
        S = []

        for j in range(dim**2):
            R = (
                tensor(dag(ket(dim**2, j)), eye(dim))
                @ (psi @ dag(psi))
                @ tensor(ket(dim**2, j), eye(dim))
            )
            p.append(Tr(R))
            rho = R / Tr(R)
            rho_out = apply_channel(K, rho)
            S.append(rho_out)

        return -np.real(Holevo_inf_ensemble(p, S))

    x_init = np.random.rand(2 * dim**3)

    opt = minimize(objfunc, x_init, options={"disp": display})

    return -opt.fun


def min_output_entropy(K, dim, display=True):
    """
    Computes the minimum output entropy of a channel given by its set of
    Kraus operators K. dim is the dimension of the input space of the
    channel.
    """

    def objfunc(x):
        Re = np.array(x[0:dim])
        Im = np.array(x[dim:])

        psi = np.array([Re + 1j * Im]).T
        psi = psi / norm(psi)

        rho = psi @ dag(psi)
        rho_out = apply_channel(K, rho)

        return entropy(rho_out)

    x_init = np.random.rand(2 * dim)

    opt = minimize(objfunc, x_init, options={"disp": display})

    return opt.fun
