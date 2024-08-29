#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2023 Sumeet Khatri.
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

import itertools

import cvxpy as cvx
import numpy as np
from cvxpy.settings import CVXOPT
from numpy.linalg import matrix_power

from qutipy.channels import diamond_norm
from qutipy.fidelities import fidelity
from qutipy.gates import CNOT_ij, Rx_i
from qutipy.general_functions import (
    Tr,
    dag,
    eye,
    ket,
    partial_trace,
    syspermute,
    tensor,
    trace_norm,
)
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy
from qutipy.pauli import generate_nQubit_Pauli_Z
from qutipy.states import bell, apply_isotropic_twirl, graph_state
from qutipy.weyl import discrete_Weyl_X, discrete_Weyl_Z, CNOT_qudit


def state_discrimination(
    S, p, err=False, dual=False, display=False, return_meas=False
):
    """
    Calculates the optimal success probability for multiple quantum state discrimination,
    where S is a list of states and p is a list of probablities, such that p[i] is
    the probability of having the state S[i]. More generally, S can be list of
    positive semi-definite operators (not necessarily unit trace).

    If err=True, then this function returns the optimal error probability instead.
    """

    d=S[0].shape[0]
    N=len(S)

    if return_meas==True:
        dual=False

    # In the binary case, use the known closed-form expression for the
    # success probability in terms of the trace norm
    if len(S)==2:
        p_succ=(1/2)*(p[0]*Tr(S[0])+p[1]*Tr(S[1])+trace_norm(p*S[0]-(1-p)*S[1]))
        if err:
            return 1-p_succ
        else:
            return p_succ
    else:
        if not dual:
            M={}

            for i in range(N):
                M[i]=cvx.Variable((d,d),hermitian=True)

            c=[M[i]>>0 for i in range(N)]+[eye(d)-M[i]>>0 for i in range(N)]

            P=M[0]
            for i in range(1,N):
                P=P+M[i]

            c+=[P==eye(d)]

            f=sum([p[i]*cvx.trace(S[i]@M[i]) for i in range(N)])

            obj = cvx.Maximize(
                cvx.real(f)
            )
            prob = cvx.Problem(obj, constraints=c)

            prob.solve(verbose=display, eps=1e-9)

            p_succ = prob.value

            if err:
                if return_meas:
                    return 1-p_succ,[M[i].value for i in range(N)]
                else:
                    return 1 - p_succ
            else:
                if return_meas:
                    return p_succ,[M[i].value for i in range(N)]
                else:
                    return p_succ

        elif dual:

            W = cvx.Variable((d, d), hermitian=True)

            c=[W>>p[i]*S[i] for i in range(N)]

            obj = cvx.Minimize(cvx.real(cvx.trace(W)))
            prob = cvx.Problem(obj, constraints=c)

            prob.solve(verbose=display, eps=1e-9)

            p_succ = prob.value

            if err:
                return 1 - p_succ
            else:
                return p_succ

### TODO: function for minimax state discrimination


def apply_teleportation_chain_channel(rho, n, dA=2, dR=2, dB=2):
    """
    Applies the teleportation chain channel to the state rho, which is of the form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}.

    The channel is defined by performing a d-dimensional Bell basis measurement
    independently on the system pairs Ri1 and Ri2, for 1 <= i <= n; based on the
    outcome, a 'correction operation' is applied to B. The system pairs Ri1 and Ri2
    can be thought of as 'repeaters'. Note that n>=1. For n=1, we get the same channel
    as in apply_teleportation_channel().

    We obtain teleportation by letting dA=1 and letting

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B} = psi_{R11} ⊗ Phi_{R12 R21}^+ ⊗ ... ⊗ Phi_{Rn2 B}^+,

    so that we have teleportation of the state psi in the system R11 to the system B.

    We obtain a chain of entanglement swaps by letting

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B} = Phi_{A R11}^+ ⊗ Phi_{R12 R21}^+ ⊗ ... ⊗ Phi_{Rn2 B}^+.
    """

    indices = list(itertools.product(*[range(dB)] * n))

    rho_out = np.array(np.zeros((dA * dB, dA * dB), dtype=complex))

    for z_indices in indices:
        for x_indices in indices:
            Bell_zx = bell(z_indices[0], x_indices[0],dB)
            for j in range(1, n):
                Bell_zx = tensor(Bell_zx, bell(z_indices[j], x_indices[j],dB))

            z_sum = np.mod(sum(z_indices), dB)
            x_sum = np.mod(sum(x_indices), dB)

            W_zx = matrix_power(discrete_Weyl_Z(dB), z_sum) @ matrix_power(
                discrete_Weyl_X(dB), x_sum
            )

            rho_out = rho_out + tensor(eye(dA), dag(Bell_zx), W_zx) @ rho @ tensor(
                eye(dA), Bell_zx, dag(W_zx)
            )

    return rho_out


def post_graph_state_dist_fidelity(A_G, n, rho):
    """
    Finds the fidelity of the output state of the graph state distribution channel
    with respect to the graph state |G>, where A_G is the adjacency matrix of the
    graph G and n is the number of vertices of G. The input state rho is a state of
    2*n qubits.
    """

    X_n = list(itertools.product(*[range(2)] * n))

    f = 0

    for x_n in X_n:
        x_n = np.array([x_n]).T  # Turn x_n into a column vector matrix

        z_n = A_G * x_n
        z_n = np.mod(z_n, 2)

        bell = bell(z_n[0, 0], x_n[0, 0], 2, as_matrix=True)

        for k in range(1, n):
            bell = tensor(bell, bell(z_n[k, 0], x_n[k, 0], 2, as_matrix=True))

        bell = syspermute(
            bell,
            list(range(1, 2 * n, 2)) + list(range(2, 2 * n + 1, 2)),
            2 * np.ones(2 * n, dtype=int),
        )

        f += fidelity(rho, bell)

    return f


def post_teleportation_fidelity(rho, dA=2):
    """
    Calculates the fidelity of the output state of the teleportation channel with
    respect to the maximally entangled state on AB. The input state rho is of the
    form rho_{AR1R2B}. We assume that A, R1, R2, B all have the same dimension.
    """

    return sum(
        [
            fidelity(
                rho,
                tensor(
                    bell(z, x, dA, as_matrix=True),
                    bell(z, x, dA, as_matrix=True),
                ),
            )
            for z in range(dA)
            for x in range(dA)
        ]
    )


def post_ent_swap_GHZ_chain_fidelity(rho, n):
    """
    Finds the fidelity of the output state of the apply_ent_swap_GHZ_chain_channel()
    function with respect to the (n+2)-party GHZ state.
    """

    indices = list(itertools.product(*[range(2)] * n))

    f = 0

    for index in indices:
        index = list(index)

        s = np.mod(sum(index), 2)

        Bell_z = bell(s, 0, 2, as_matrix=True)

        for z in index:
            Bell_z = tensor(Bell_z, bell(z, 0, 2, as_matrix=True))

        f = f + fidelity(Bell_z, rho)

    return f


def apply_ent_swap_GHZ_channel(rho):
    """
    Applies the channel that takes two copies of a maximally entangled state and outputs
    a three-party GHZ state. The input state rho is of the form

        rho_{A R1 R2 B}.

    A CNOT is applied to R1 and R2, followed by a measurement in the standard basis on
    R2, followed by a correction operation on B based on the outcome of the measurement.

    Currently only works for qubits.
    """

    C = CNOT_ij(2, 3, 4)

    X = [matrix_power(discrete_Weyl_X(2), x) for x in range(2)]

    rho_out = np.sum(
        [
            tensor(eye(4), dag(ket(2, x)), eye(2))
            @ C
            @ tensor(eye(8), X[x])
            @ rho
            @ tensor(eye(8), X[x])
            @ dag(C)
            @ tensor(eye(4), ket(2, x), eye(2))
            for x in range(2)
        ],
        0,
    )

    return rho_out


def apply_ent_swap_GHZ_chain_channel(rho, n):
    """
    Applies the channel that takes n+1 copies of a maximally entangled state and outputs
    a (n+2)-party GHZ state. The input state rho is of the form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}

    A CNOT is applies to each pair Rj1 Rj2. Then, the qubits Rj2 are measured in the
    standard basis. Conditioned on these outcomes, a correction operation is applied
    at B.

    Currently only works for qubits. For n=1, we get the same thing as apply_ent_swap_GHZ_channel().
    """

    def K(j, x):
        # j is between 1 and n, denoting the pair of R systems. x is either 0 or 1.
        # For each j, the qubit indices are 2*j and 2*j+1 for the pair Rj1 and
        # Rj2

        Mx = tensor(
            eye(2),
            eye(2 ** (2 * j - 2)),
            eye(2),
            ket(2, x) @ dag(ket(2, x)),
            eye(2 ** (2 * (n - j))),
            eye(2),
        )

        C = CNOT_ij(2 * j, 2 * j + 1, 2 * n + 2)

        X = 1j * Rx_i(2 * j + 2, np.pi, 2 * n + 2)

        return Mx @ C @ matrix_power(X, x)

    indices = list(itertools.product(*[range(2)] * n))

    rho_out = np.array(np.zeros((2 ** (2 * n + 2), 2 ** (2 * n + 2)), dtype=complex))

    for index in indices:
        index = list(index)

        L = K(1, index[0])
        for j in range(2, n + 1):
            L = K(j, index[j - 1]) @ L

        rho_out = rho_out + L @ rho @ dag(L)

    rho_out = partial_trace(
        rho_out, [2 * j + 1 for j in range(1, n + 1)], [2] * (2 * n + 2)
    )

    return rho_out


def channel_discrimination(
    C, dA, dB, P, err=False, dual=False, display=False
):
    """
    Calculates the optimal success probability for multiple quantum channel
    discrimination, in the Bayesian setting. The list C contains the Choi
    representations of the channels, and the list p contain the prior probabilities,
    with p[i] being the probability of having the channel C[i]. dA and dB are
    the input and output dimensions, respectively, of the channels.

    If err=True, then this function returns the optimal error probability instead.
    """
    
    if len(C)==2:
        p_succ = (1 / 2) * (
            1
            + diamond_norm(
                P[0] * C[0] - P[1] * C[1], dA, dB, display=display, choi=True
            )
        )
        if err:
            return 1 - p_succ
        else:
            return p_succ

    else:

        n=len(C)

        if not dual:

            Q={}
            for i in range(n):
                Q[i]=cvx.Variable((dA*dB,dA*dB),hermitian=True)

            sigma=cvx.Variable((dA,dA),hermitian=True)

            c=[Q[i]>>0 for i in range(n)]+[sigma>>0,cvx.trace(sigma)==1]
            c+=[cvx.sum([Q[i] for i in range(n)])==cvx.kron(sigma,eye(dB))]

            obj=cvx.Maximize(cvx.sum([P[i]*cvx.real(cvx.trace(Q[i]@C[i])) for i in range(n)]))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-9)

            p_succ=prob.value

            if err:
                return 1 - p_succ
            else:
                return p_succ

        elif dual:

            l = cvx.Variable()
            Y = cvx.Variable((dA * dB, dA * dB), hermitian=True)

            YA = numpy_to_cvxpy(partial_trace(cvxpy_to_numpy(Y), [2], [dA, dB]))

            c=[l>=0]+[Y>>P[i]*C[i] for i in range(n)]+[Y>>0,YA==l*eye(dA)]

            obj = cvx.Minimize(l)
            prob = cvx.Problem(obj, constraints=c)

            prob.solve(verbose=display, eps=1e-9)

            p_succ = prob.value

            if err:
                return 1 - p_succ
            else:
                return p_succ


def channel_discrimination_minimax(C,dA,dB,err=False, dual=False, display=False):

    """
    Calculates the optimal success probability for multiple quantum channel
    discrimination, in the minimax setting. The list C contains the Choi
    representations of the channels. dA and dB are the input and output
    dimensions, respectively, of the channels.

    If err=True, then this function returns the optimal error probability instead.
    """

    n=len(C)

    if not dual:

        q=cvx.Variable()
        P={}
        for i in range(n):
            P[i]=cvx.Variable((dA*dB,dA*dB),hermitian=True)
        sigma=cvx.Variable((dA,dA),hermitian=True)

        c=[q>=0]+[q<=cvx.real(cvx.trace(P[i]@C[i])) for i in range(n)]+[P[i]>>0 for i in range(n)]
        c+=[cvx.sum([P[i] for i in range(n)])==cvx.kron(sigma,eye(dB))]
        c+=[sigma>>0,cvx.trace(sigma)==1]

        obj=cvx.Maximize(q)
        prob=cvx.Problem(obj,constraints=c)

        prob.solve(verbose=display,eps=1e-7)

        p_succ=prob.value

        if err:
            return 1-p_succ
        else:
            return p_succ        

    else:

        l=cvx.Variable()
        P=cvx.Variable(n)
        Y=cvx.Variable((dA*dB,dA*dB),hermitian=True)

        YA=numpy_to_cvxpy(partial_trace(cvxpy_to_numpy(Y),[2],[dA,dB]))

        c=[l>=0]+[Y>>P[i]*C[i] for i in range(n)]+[p>=0 for p in P]+[cvx.sum(P)==1]+[Y>>0,YA==l*eye(dA)]

        obj=cvx.Minimize(l)

        prob=cvx.Problem(obj,constraints=c)

        prob.solve(verbose=display,eps=1e-7)

        p_succ=prob.value

        if err:
            return 1-p_succ
        else:
            return p_succ



def post_ent_swap_GHZ_fidelity(rho):
    """
    Finds the fidelity of the output state of the apply_ent_swap_GHZ_channel() function
    with respect to the three-party GHZ state.
    """

    Phi = [bell(z, 0, 2, as_matrix=True) for z in range(2)]

    return sum([fidelity(tensor(Phi[z], Phi[z]), rho) for z in range(2)])


def entanglement_distillation(
    rho1, rho2, d, outcome=1, twirl_before=False, twirl_after=False, normalize=True, return_prob=True
):
    """
    Applies a particular entanglement distillation channel to the two two-qubit states
    rho1 and rho2. [PRL 76, 722 (1996); see also PRL 77, 2818 (1996)] (if d=2).
    
    If d>2, then we apply an analogous protocol, which can be
    found in [PRA 59, 4206 (1999); see also J. Phys. A: Math. Gen. 34, 8821 (2001)].
    In this case, rho1 and rho2 are two two-qudit states.

    The channel is probabilistic. If the variable outcome=1, then the function returns
    the two-qubit state conditioned on the success of the distillation protocol.
    """

    if d==2:
        CNOT=CNOT_ij(3,4,4)@CNOT_ij(1,2,4)
    else:
        CNOT=tensor(CNOT_qudit(d),CNOT_qudit(d))

    K={}

    for i in range(d):
        for j in range(d):
            K[(i,j)]=tensor(eye(d),dag(ket(d,i)),eye(d),dag(ket(d,j)))@CNOT
    
    #K00=tensor(eye(2),dag(ket(2,0)),eye(2),dag(ket(2,0)))@CNOT
    #K01=tensor(eye(2),dag(ket(2,0)),eye(2),dag(ket(2,1)))@CNOT
    #K10=tensor(eye(2),dag(ket(2,1)),eye(2),dag(ket(2,0)))@CNOT
    #K11=tensor(eye(2),dag(ket(2,1)),eye(2),dag(ket(2,1)))@CNOT

    if twirl_before:
        rho1=apply_isotropic_twirl(rho1,d)
        rho2=apply_isotropic_twirl(rho2,d)

    rho_in = syspermute(
        tensor(rho1, rho2), [1, 3, 2, 4], [d, d, d, d]
    )  # rho_in==rho_{A1A2B1B2}

    if outcome == 1:  ## When the protocol is successful
        # rho_out is unnormalized. The trace of rho_out is equal to the success
        # probability.
        
        rho_out=np.sum([K[(i,i)]@rho_in@dag(K[(i,i)]) for i in range(d)],0)

        if return_prob:
            prob=Tr(rho_out)
        
        if twirl_after:
            rho_out = apply_isotropic_twirl(rho_out, d)
        if normalize:
            rho_out = rho_out / Tr(rho_out)

    elif outcome == 0:
        # rho_out is unnormalized. The trace of rho_out is equal to the failure
        # probability.
        
        rho_out=np.sum([K[(i,j)]@rho_in@dag(K[(i,j)]) for i in range(d) for j in range(d) if i!=j],0)

        if return_prob:
            prob=Tr(rho_out)
        
        if normalize:
            rho_out = rho_out / Tr(rho_out)

    if return_prob:
        return rho_out,prob
    else:
        return rho_out


def apply_graph_state_dist_channel(A_G, n, rho):
    """
    Applies the graph state distribution channel to the 2n-partite state rho, where
    n is the number of vertices in the graph G with adjacency matrix A_G (binary
    symmetric matrix).

    rho is a state of the form rho_{A_1...A_n R_1...R_n}

    The local graph state operations and measurements are applied to the qubits
    R_1,...,R_n, and the correction operations are applied to A_1,...,A_n.

    When rho is a state of the form

        Phi_{A_1 R_1}^+ ⊗ Phi_{A_2 R_2}^+ ⊗ ... ⊗ Phi_{A_n R_n}^+,

    then the output state on A_1,...,A_n is the graph state |G>.
    """

    indices = list(itertools.product(*[range(2)] * n))

    ket_G = graph_state(A_G, n)

    rho_out = np.array(np.zeros((2 ** (2 * n), 2 ** (2 * n)), dtype=complex))

    for index in indices:
        Zx = generate_nQubit_Pauli_Z(index)

        Gx = Zx * ket_G
        rho_out = rho_out + tensor(Zx, dag(Gx)) @ rho @ tensor(Zx, Gx)

    return rho_out


def apply_teleportation_channel(rho, dA=2, dR1=2, dR2=2, dB=2):
    """
    Applies the d-dimensional teleportation channel to the four-qudit state rho_{AR1R2B}.
    The channel measures R1 and R2 in the d-dimensional Bell basis and, based on the
    outcome, applies a 'correction operation' to B. So the output of the channel consists
    only of the systems A and B.

    We obtain quantum teleportation by letting

        rho_{AR1R2B} = psi_{R1} ⊗ Phi_{R2B}^+,

    so that dA=1. This simulates teleportation of the state psi in the system R1 to
    the system B.

    We obtain entanglement swapping by letting

        rho_{AR1R2B} = Phi_{AR1}^+ ⊗ Phi_{R2B}^+.

    The result of the channel is then Phi_{AB}^+
    """

    X = [matrix_power(discrete_Weyl_X(dB), x) for x in range(dB)]
    Z = [matrix_power(discrete_Weyl_Z(dB), z) for z in range(dB)]

    rho_out = np.sum(
        [
            tensor(eye(dA), dag(bell(z, x, dR1)), Z[z] @ X[x])
            @ rho
            @ tensor(eye(dA), bell(z, x, dR1), dag(X[x]) @ dag(Z[z]))
            for z in range(dB)
            for x in range(dB)
        ],
        0,
    )

    return rho_out


def post_teleportation_chain_fidelity(rho, n, dA=2):
    """
    Calculates the fidelity of the output state of the teleportation chain channel with
    respect to the maximally entangled state on AB. The input state rho is of the
    form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}.

    We assume that A, B, and all R systems have the same dimension.
    """

    f = 0

    indices = list(itertools.product(*[range(dA)] * n))

    for z_indices in indices:
        for x_indices in indices:
            z_sum = np.mod(sum(z_indices), dA)
            x_sum = np.mod(sum(x_indices), dA)

            Bell_tot = bell(z_sum, x_sum, dA, as_matrix=True)

            for j in range(n):
                Bell_tot = tensor(
                    Bell_tot,
                    bell(z_indices[j], x_indices[j], dA, as_matrix=True),
                )

            #f += fidelity(rho, Bell_tot)
            f+=Tr(Bell_tot@rho)

    return f
