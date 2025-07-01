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

import numpy as np
from numpy.linalg import matrix_power, norm, inv

import itertools

from math import factorial
from sympy.combinatorics.permutations import Permutation

from qutipy.gates import CZ_ij
from qutipy.general_functions import SWAP, Tr, dag, eye, ket, syspermute, tensor
from qutipy.pauli import generate_nQubit_Pauli
from qutipy.weyl import discrete_Weyl, discrete_Weyl_X, discrete_Weyl_Z, discrete_Weyl_basis
from qutipy.linalg import Sqrtm, vec, vec_inverse


def max_ent(dim, normalized=True, as_matrix=True):
    """
    Generates the dim-dimensional maximally entangled state, which is defined as

    (1/sqrt(dim))*(|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If normalized=False, then the function returns the unnormalized maximally entangled
    vector.

    If as_matrix=True, then the function returns the state as a density matrix.
    """

    if normalized:
        Bell = (1.0 / np.sqrt(dim)) * np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if as_matrix:
            return Bell @ dag(Bell)
        else:
            return Bell
    else:
        Gamma = np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if as_matrix:
            return Gamma @ dag(Gamma)
        else:
            return Gamma


def bell(z, x, d=2, as_matrix=False, normalized=True, n_qubit=False):
    """
    Generates a d-dimensional Bell state with 0 <= z,x <= d-1. These are defined as

    |Phi_{z,x}> = (Z(z)X(x) ⊗ I)|Phi^+>

    If n_qubit=True, then we should have that d=2^n, such that we can define 
    the Bell states in terms of the n-qubit Pauli operators instead. Then
    z and x should be lists, each of size n, each consisting of zeros 
    and ones. 
    """

    Bell = max_ent(d, as_matrix=as_matrix, normalized=normalized)

    if not n_qubit:
        W_zx = matrix_power(discrete_Weyl_Z(d), z) @ matrix_power(discrete_Weyl_X(d), x)

        if as_matrix:
            return tensor(W_zx, eye(d)) @ Bell @ tensor(dag(W_zx), eye(d))
        else:
            return tensor(W_zx, eye(d)) @ Bell
    else:
        n=int(np.log2(d))  # The number of qubits
        
        P_zx=generate_nQubit_Pauli([z,x],alt=True)

        if as_matrix:
            return tensor(P_zx,eye(d))@Bell@tensor(dag(P_zx),eye(d))
        else:
            return tensor(P_zx,eye(d))@Bell


def generate_Bell_basis(d,n_qubit=False,linop=False,as_dict=False):
    '''
    Generates a list of all two-qudit Bell states, where
    d is the dimension of each qudit.

    If n_qubit=True, then we should have d=2^n, such that
    we can use the n-qubit Pauli operators to generate 
    the Bell states.

    If linop=True, then this generates a list of operators
    corresponding to the basis of linear operators acting
    on two qudits.
    '''

    if as_dict:
        B={}
    else:
        B=[]

    if n_qubit:
        n=int(np.log2(d))  # The number of qubits
        S=list(itertools.product([0, 1], repeat=n))
    else:
        S=list(range(d))
    
    for s1 in S:
        for s2 in S:
            if as_dict:
                B[(s1,s2)]=bell(s1,s2,d,n_qubit=n_qubit)
            else:
                B.append(bell(s1,s2,d,n_qubit=n_qubit))

    if not linop:
        return B
    else:
        if as_dict:
            Bop={}
            for s1 in S:
                for s2 in S:
                    for s3 in S:
                        for s4 in S:
                            Bop[(s1,s2,s3,s4)]=B[(s1,s2)]@dag(B[(s3,s4)])
            return Bop
        else:
            return [b1@dag(b2) for b1 in B for b2 in B]


def Bell_diagonal_state(d,p,n_qubit=False):
    """
    Generates a two-qudit Bell-diagonal state, with local dimension d.

    If n_qubit=True, then we should have d=2^n, and then we use the
    Bell states defined by the n-qubit Pauli operators.

    The variable p is a dictionary of d^2 probabilities, specified
    in the form p[(s1,s2)]. Note that they can be arbitrary real 
    numbers -- they need not be probabilities, in case we want a 
    Hermitian Bell-diagonal operator.
    """

    if n_qubit:
        n=int(np.log2(d))  # The number of qubits
        S=list(itertools.product([0, 1], repeat=n))
    else:
        S=list(range(d))

    return np.sum([p[(s1,s2)]*bell(s1,s2,d,as_matrix=True,n_qubit=n_qubit) for s1 in S for s2 in S],0)


def random_Bell_diagonal(d,PSD=True,normalized=True):

    index_set=list(itertools.product(range(d),repeat=2))

    if PSD==True and normalized==True:
        p=random_probability_distribution(d**2,as_dict=True,index_set=index_set)
        return Bell_diagonal_state(d,p)

    elif PSD==True and normalized==False:
        p=dict(zip(index_set,np.random.rand(d**2)))
        return Bell_diagonal_state(d,p)
    
    else:
        p=dict(zip(index_set,np.random.randn(d**2)))
        return Bell_diagonal_state(d,p)


def GHZ(dim, n, as_matrix=True):
    """
    Generates the n-party GHZ state in dim-dimensions for each party, which is defined as

        |GHZ_n> = (1/sqrt(dim))*(|0,0,...,0> + |1,1,...,1> + ... + |d-1,d-1,...,d-1>)

    If as_matrix=True, then the function returns the state as a density matrix.
    """

    GHZ = (1 / np.sqrt(dim)) * np.sum([ket(dim, [i] * n) for i in range(dim)], 0)

    if as_matrix:
        return GHZ @ dag(GHZ)
    else:
        return GHZ


def graph_state(A_G, n, as_matrix=False, return_CZ=False, alt=True):
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

    if as_matrix:
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

    Bell = max_ent(d)

    if fidelity:
        return p * Bell + ((1 - p) / (d**2 - 1)) * (eye(d**2) - Bell)
    else:
        return p * Bell + (1 - p) * eye(d**2) / d**2


def apply_isotropic_twirl(X, d):
    """
    Applies the twirling channel

        X -> ∫ (U ⊗ conj(U))*X*(U ⊗ conj(U))^† dU

    to the input operator X acting on two d-dimensional systems.

    For d=2, this is equivalent to

        X -> (1/24)*sum_i (c_i ⊗ conj(c_i))*X*(c_i ⊗ conj(c_i))^†

    where the unitaries c_i form the one-qubit Clifford group (because the Clifford
    unitaries constitute a unitary 2-design).

    This channel takes any state rho and converts it to an isotropic state with
    the same fidelity to the maximally entangled state as rho.
    """

    G = max_ent(d, normalized=False, as_matrix=True)

    return (Tr(X) / (d**2 - 1) - Tr(G @ X) / (d * (d**2 - 1))) * eye(d**2) + (
        Tr(G @ X) / (d**2 - 1) - Tr(X) / (d * (d**2 - 1))
    ) * G


def max_mix(dim):
    """
    Generates the dim-dimensional maximally mixed state.
    """

    return eye(dim) / dim


def random_density_matrix(dim, *args):
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


def random_state_vector(dim, rank=None, as_matrix=False):
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

        if as_matrix:
            return psi @ dag(psi)
        else:
            return psi
    else:
        dimA = dim[0]
        dimB = dim[1]

        if rank is None:
            rank = max([dimA, dimB])
        else:
            k = rank

        psi_k = max_ent(k, as_matrix=False, normalized=False)
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

        if as_matrix:
            return psi @ dag(psi)
        else:
            return psi
        

def random_probability_distribution(d,as_dict=False,index_set=None):
    """
    Generates a random (discrete) probability distribution on d elements.
    Makes use of the random density matrix function.
    """

    rho=random_density_matrix(d)

    if as_dict:
        if index_set==None:
            index_set=range(1,d+1)
        return dict(zip(index_set,np.real(np.diag(rho))))
    else:
        return np.real(np.diag(rho))



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


def apply_Werner_twirl(X, d):
    """
    Applies the twirling channel

        X -> ∫ (U ⊗ U)*rho*(U ⊗ U)^† dU

    to the input operator X acting on two d-dimensional systems.

    For d=2, this is equivalent to

        X -> (1/24)*sum_i (c_i ⊗ c_i)*X*(c_i ⊗ c_i)^†

    where the unitaries c_i form the one-qubit Clifford group (because the Clifford
    unitaries constitute a unitary 2-design).

    This channel takes any state rho and converts it to a Werner state with
    the same fidelity to the singlet state as rho.
    """

    F = SWAP([1, 2], [d, d])

    return (Tr(X) / (d**2 - 1) - Tr(F @ X) / (d * (d**2 - 1))) * eye(d**2) + (
        Tr(F @ X) / (d**2 - 1) - Tr(X) / (d * (d**2 - 1))
    ) * F


def apply_discrete_Weyl_twirl(X, d, n):
    """
    Applies a discrete Weyl twirling channel to the input operator X.
    The number n is the number of subsystems, and d is the local dimension.
    So the operator X acts on the vector space (C^d)^{⊗ n}

    For example, if n=2, and accordingly X is a bipartite operator, then
    the twirling channel is

        X -> \sum_{z,x=0}^{d-1} (W_{z,x} ⊗ W_{z,x}) X (W_{z,x} ⊗ W_{z,x})^†

    For d=2, this is the same as the Pauli twirl -- see the 'apply_Pauli_twirl' function.
    """

    return (1/d**2)*np.sum(
        [
            tensor([discrete_Weyl(d, z, x), n])
            @ X
            @ tensor([dag(discrete_Weyl(d, z, x)), n])
            for z in range(d)
            for x in range(d)
        ],
        0,
    )


def apply_Pauli_twirl(X, n, m=1,alt=False):
    """
    Applies a Pauli twirl to an operator X acting on a system of n*m-qubits.
    So the operator X acts on the vector space (C^{2^m})^{⊗ n}.

    For example, if n=3, then the twirling channel is

        X -> \sum_{i=0}^4 (P_i ⊗ P_i ⊗ P_i) X (P_i ⊗ P_i ⊗ P_i)
    """

    if alt:
        x=list(itertools.product([0, 1], repeat=m))
        z=list(itertools.product([0, 1], repeat=m))

        return np.sum(
            [
                generate_nQubit_Pauli([z*n,x*n],alt=True) @ X @ dag(generate_nQubit_Pauli([z*n,x*n],alt=True))
                for s in S
            ],
            0,
        )
    else:
        S = list(itertools.product([0, 1, 2, 3], repeat=m))

        return np.sum(
            [
                generate_nQubit_Pauli(s * n) @ X @ dag(generate_nQubit_Pauli(s * n))
                for s in S
            ],
            0,
        )


def purification(rho,as_matrix=False,alt=False):
    """
    Returns the 'canonical purification' of a density operator rho, defined as

        |psi_rho>=(I ⊗ sqrt(rho))*|Gamma>,
    
    where |Gamma> is the (unnormalized) maximally-entangled vector. Here, the
    first tensor factor represents the 'purifying system'.
    
    If alt=True, then we take instead

        |psi_rho>=(sqrt(rho) ⊗ I)*|Gamma>,
    
    such that the second tensor factor represents the purifying system.
    """

    d=rho.shape[0]

    Gamma=max_ent(d,normalized=False,as_matrix=as_matrix)
    rho_sq=Sqrtm(rho)

    if alt:
        R=tensor(rho_sq,eye(d))
    else:
        R=tensor(eye(d),rho_sq)

    if as_matrix:
        return R@Gamma@dag(R)
    else:
        return R@Gamma


def occupation_number_state_sym(d,n,c):
    """
    Returns an occupation number state corresponding to the 
    symmetric subspace of n tensor copies of C^d The variable
    c is a list/tuple of d integers between 0 and d-1, which
    should sum up to n.
    """

    if np.sum(c)!=n:
        return "Sum of the occupation numbers should be equal to n."

    v=1
    
    for i in range(d):
        v=tensor(v,[ket(d,i),c[i]])

    perms=list(itertools.permutations(list(range(1,n+1))))

    out=np.zeros((d**n,1),dtype=np.complex128)

    for perm in perms:
        out=out+syspermute(v,perm,[d]*n)

    N=(1/np.sqrt(factorial(n)))*(1/np.sqrt(np.prod([factorial(ci) for ci in c])))

    return N*out


def occupation_number_state_asym(d,n,c):
    """
    Returns an occupation number state corresponding to the 
    anti-symmetric subspace of n tensor copies of C^d The variable
    c is a list/tuple of d integers between 0 and d-1, which
    should sum up to n.
    """

    if np.sum(c)!=n:
        return "Sum of the occupation numbers should be equal to n."

    v=1
    
    for i in range(d):
        v=tensor(v,[ket(d,i),c[i]])

    perms=list(itertools.permutations(list(range(1,n+1))))

    out=np.zeros((d**n,1),dtype=np.complex128)

    for perm in perms:
        sign=Permutation([p-1 for p in perm]).signature()
        out=out+sign*syspermute(v,perm,[d]*n)

    N=(1/np.sqrt(factorial(n)))

    return N*out

############################################################################

# QuTIpy States Utility

import cvxpy as cvx

from qutipy.general_functions import partial_trace, partial_transpose, trace_norm


def density_matrix_basis(d,return_dual=False):
    """
    Returns an operator basis for d dimensions consisting of density operators.
    Note that all the density operators in this construction are rank-one
    projections.

    If return_dual=True, then the function also returns a dual operator basis.
    """

    R=[]

    for a in range(d):
        for b in range(d):
            if a==b:
                R.append(ket(d,a)@dag(ket(d,a)))
            elif a<b:
                p=(1/np.sqrt(2))*(ket(d,a)+ket(d,b))
                R.append(p@dag(p))
            elif a>b:
                p=(1/np.sqrt(2))*(ket(d,a)+1j*ket(d,b))
                R.append(p@dag(p))
    
    if return_dual:
        F=np.sum([vec(rho)@dag(vec(rho)) for rho in R],0)
        O=[vec_inverse(inv(F)@vec(rho),d,d) for rho in R]
        return R, O
    else:
        return R


def density_matrix_POVM(d,return_dual=False):
    """
    Returns an information-complete POVM based on the basis of density
    operators in the function density_matrix_basis.

    If return_dual=True, then the function also returns a dual operator
    basis.
    """

    R=density_matrix_basis(d)

    Q=np.sum([r for r in R],0)

    M=[Sqrtm(inv(Q))@r@Sqrtm(inv(Q)) for r in R]

    if return_dual:
        F=np.sum([vec(m)@dag(vec(m)) for m in M],0)
        O=[vec_inverse(inv(F)@vec(m),d,d) for m in M]
        return M, O
    else:
        return M
    

def discrete_Weyl_POVM(d,rho,return_dual=False):
    """
    Generates a POVM (positive operator-valued measure) in a d-dimensional
    space using the discrete-Weyl operators. Here, rho can be an arbitrary 
    quantum state.

    This POVM is information-complete as long as Tr(rho * w) ≠ 0 for all
    elements w of the discrete-Weyl basis.

    If return_dual=True, then this returns the (unique) dual frame.

    For more information, we refer to:
        'Informationlly complete measurements and group representation',
        G. M. D'Ariano et al., J. Opt. B: Quantum Semiclass. Opt. 6, S487 (2004).
    """

    M=[(1/d)*w@rho@dag(w) for w in discrete_Weyl_basis(d)]

    if return_dual:
        F=np.sum([vec(m)@dag(vec(m)) for m in M],0)
        O=[vec_inverse(inv(F)@vec(m),d,d) for m in M]
        return M,O
    else:
        return M


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
