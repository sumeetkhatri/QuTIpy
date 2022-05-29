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

import itertools

import cvxpy
import numpy as np
from numpy.linalg import norm

from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy


def dag(X):
    """
    Takes the numpy array X and returns its complex conjugate transpose.
    """

    return X.conj().T


def eye(n):
    """
    Generates the n x n identity matrix.
    """

    return np.identity(n, dtype=float)


def generate_all_kets(dims):
    """
    Generates the tensor-product orthonormal basis corresponding to vector spaces
    with dimensions in the list dims.

    ------------------------
    Example:

        generate_all_kets([2,3]) returns a list containing

            |0,0>
            |0,1>
            |0,2>
            |1,0>
            |1,1>
            |1,2>

    """

    dims_set = [range(d) for d in dims]

    L = list(itertools.product(*dims_set))

    K = []

    for l in L:
        K.append(ket(dims, l))

    return K


def get_subblock(X, sys, indices, dim):
    """For the multipartite operator X, with local dimensions given by the list dim,
    this function extracts the subblock such that the systems in sys have the
    values given by those in indices. indices should be a list of tuples, each
    tuple corresponding to a system in sys. For each tuple, the first element
    gives the row and the second the column.

    For example, the subblock (<i| ⊗ id ⊗ <j|)X(|k> ⊗ id ⊗ |l>)
    is given by get_subblock(X,[1,3],[(i,k),(j,l)],[dim1,dim2,dim3]), so the result
    is a dim2 x dim2 matrix, which is the desired subblock.


    Args:
        X (_type_): _description_
        sys (_type_): _description_
        indices (_type_): _description_
        dim (_type_): _description_. By default it will tak the dimensions from X.

    Returns:
        _type_: _description_
    """

    X = np.array(X)

    X_reshape = np.reshape(X, dim + dim)

    num_sys = len(dim)
    total_sys = range(1, num_sys + 1)
    dims_keep = [dim[s - 1] for s in list(set(total_sys) - set(sys))]

    to_slice = [slice(None) for i in range(2 * num_sys)]

    count = 0
    for s in total_sys:
        if s in sys:
            to_slice[s - 1] = indices[count][0]
            to_slice[num_sys + s - 1] = indices[count][1]
            count += 1

    X_reshape = X_reshape[tuple(to_slice)]

    X_reshape = np.reshape(X_reshape, dims_keep + dims_keep)

    return X_reshape


def ket(dim, *args):
    """
    Generates a standard basis vector in dimension dim.

    For example, ket(2,0)=|0> and ket(2,1)=|1>.

    In general, ket(d,j), for j between 0 and d-1, generates a column vector
    (as a numpy matrix) in which the jth element is equal to 1 and the rest
    are equal to zero.

    ket(d,[j1,j2,...,jn]) generates the tensor product |j1>|j2>...|jn> of
    d-dimensional basis vectors.

    If dim is specified as a list, then, e.g., ket([d1,d2],[j1,j2]) generates the
    tensor product |j1>|j2>, with the first tensor factor being d1-dimensional
    and the second tensor factor being d2-dimensional.
    """

    args = np.array(args)

    if args.size == 1:
        num = args[0]
        out = np.zeros([dim, 1])
        out[num] = 1
    else:
        args = args[0]
        if isinstance(dim, int):
            out = ket(dim, args[0])
            for j in range(1, len(args)):
                out = np.kron(out, ket(dim, args[j]))
        elif isinstance(dim, list):
            out = ket(dim[0], args[0])
            for j in range(1, len(args)):
                out = np.kron(out, ket(dim[j], args[j]))

    return out


def partial_trace(X, sys, dim):
    """
    sys is a list of systems over which to take the partial trace (i.e., the
    systems to discard).

    Example: If rho_AB is a bipartite state with dimA the dimension of system A
    and dimB the dimension of system B, then

    partial_trace(rho_AB,[2],[dimA,dimB]) gives the density matrix on

    system A, i.e., rho_A:=partial_trace[rho_AB].

    Similarly, partial_trace(rho_AB,[1],[dimA,dimB]) discards the first subsystem,
    returning the density matrix of system B.

    If rho_ABC is a tripartite state, then, e.g.,

    partial_trace(rho_ABC,[1,3],[dimA,dimB,dimC])

    discards the first and third subsystems, so that we obtain the density
    matrix for system B.

    """

    if isinstance(X, cvxpy.Variable):
        X = cvxpy_to_numpy(X)
        X_out = partial_trace(X, sys, dim)
        return numpy_to_cvxpy(X_out)

    if not sys:  # If sys is empty, just return the original operator
        return X
    elif len(sys) == len(dim):  # If tracing over all systems
        return Tr(X)
    else:

        if X.shape[1] == 1:
            X = X @ dag(X)

        num_sys = len(dim)
        total_sys = range(1, num_sys + 1)

        dims_sys = [
            dim[s - 1] for s in sys
        ]  # Dimensions of the system to be traced over
        dims_keep = [dim[s - 1] for s in list(set(total_sys) - set(sys))]
        dim_sys = np.product(dims_sys)
        dim_keep = np.product(dims_keep)

        perm = sys + list(set(total_sys) - set(sys))
        X = syspermute(X, perm, dim)

        X = np.array(X)
        dim = [dim_sys] + dims_keep
        X_reshape = np.reshape(X, dim + dim)
        X_reshape = np.sum(np.diagonal(X_reshape, axis1=0, axis2=len(dim)), axis=-1)
        X = np.reshape(X_reshape, (dim_keep, dim_keep))

        return X


def partial_transpose(X, sys, dim):
    """Takes the partial transpose on systems given by sys. dim is a list of
    the dimensions of each of the subsystems in X.

    Example: If rho_AB is a bipartite state with dimA the dimension of system A
    and dimB the dimension of system B, then

    Tx(rho_AB,[2],[dimA,dimB]) takes the transpose on system B.

    Similarly, Tx(rho_AB,[1],[dimA,dimB]) takes the transpose on system A.

    If rho_ABC is a tripartite state, then, e.g.,

    TrX(rho_ABC,[1,3],[dimA,dimB,dimC])

    takes the transpose on systems A and C.

    This also works for non-square matrices provided dim is a list of tuples,
    with the first element of the tuple specifying the dimension of the rows
    of each subsystem and the second element of the tuple specifying the
    dimension of the columns of each subsystem; e.g., dim=[(2,4),(3,5)] means
    that the first tensor factor lives in an operator space taking a
    four-dimensional space to a two-dimensional space, and the second tensor
    factor lives in an operator space taking a 5-dimensional space to a
    3-dimensional space.

    Args:
        X (matrix): _description_
        sys (matrix): _description_
        dim (array): _description_

    Returns:
        _type_: _description_
    """

    if isinstance(X, cvxpy.Variable):
        X = cvxpy_to_numpy(X)
        X_out = partial_transpose(X, sys, dim)
        return numpy_to_cvxpy(X_out)

    if X.shape[1] == 1:
        X = X @ dag(X)

    X = np.array(X)

    n = len(dim)  # Number of subsystems in the operator

    if isinstance(dim[0], tuple) or isinstance(
        dim[0], list
    ):  # When the operator is a non-square matrix
        dim_row = [dim[i][0] for i in range(n)]
        dim_col = [dim[i][1] for i in range(n)]
    elif isinstance(dim[0], int):  # When the operator is a square matrix
        dim_row = dim
        dim_col = dim

    X_reshape = np.reshape(X, dim_row + dim_col)

    axes = list(range(2 * n))

    for i in range(len(sys)):
        axes[sys[i] - 1], axes[n + sys[i] - 1] = axes[n + sys[i] - 1], axes[sys[i] - 1]
        if isinstance(dim[0], tuple) or isinstance(dim[0], list):
            dim[sys[i] - 1] = list(dim[sys[i] - 1])
            dim[sys[i] - 1][0], dim[sys[i] - 1][1] = (
                dim[sys[i] - 1][1],
                dim[sys[i] - 1][0],
            )
            # dim[sys[i]-1]=tuple(dim[sys[i]-1])
        else:
            continue

    X_reshape = np.transpose(X_reshape, tuple(axes))

    if isinstance(dim[0], tuple) or isinstance(dim[0], list):
        dim_row = [dim[i][0] for i in range(n)]
        dim_col = [dim[i][1] for i in range(n)]
        dim_total = (np.product(dim_row), np.product(dim_col))
    elif isinstance(dim[0], int):
        dim_row = dim
        dim_col = dim
        dim_total = (np.product(dim), np.product(dim))

    X_new = np.reshape(X_reshape, dim_total)

    return X_new


def permute_tensor_factors(perm, dims):
    """
    Generates the permutation operator that permutes the tensor factors according
    to the given permutation.

    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    tensor factors.
    """

    K = generate_all_kets(dims)

    dim = np.prod(dims)

    W = np.zeros((dim, dim), dtype=complex)

    for ket in K:
        W = W + syspermute(ket, perm, dims) @ dag(ket)

    return W


def spectral_norm(X):
    """
    Finds the spectral norm (also known as the operator norm and the Schatten
    infinity-norm) of the matrix X. (The largest singular value of X.)
    """

    return norm(X, ord=2)


def SWAP(sys, dim):
    """
    Generates a swap matrix between the pair of systems in sys. dim is a list
    of the dimensions of the subsystems.

    For example, SWAP([1,2],[2,2]) generates the two-qubit swap matrix.
    """

    dim_total = np.product(dim)

    n = len(dim)
    sys_rest = list(np.setdiff1d(range(1, n + 1), sys))
    perm = sys + sys_rest
    p = {}

    for i in range(1, n + 1):
        p[i] = perm[i - 1]

    p2 = {v: k for k, v in p.items()}

    perm_rearrange = list(p2.values())

    dim1 = dim[sys[0] - 1]  # Dimension of the first subsystem to be swapped
    dim2 = dim[sys[1] - 1]  # Dimension of the second subsystem to be swapped

    dim_rest = int(float(dim_total) / float(dim1 * dim2))

    G1 = np.array(np.sum([ket(dim1, [i, i]) for i in range(dim1)], 0))
    G2 = np.array(np.sum([ket(dim2, [i, i]) for i in range(dim2)], 0))

    G = G1 @ dag(G2)

    S = partial_transpose(G, [2], [(dim1, dim2), (dim1, dim2)])

    P = tensor(S, eye(dim_rest))

    p_alt = list(np.array(list(p.values())) - 1)

    P = syspermute(P, perm_rearrange, list(np.array(dim)[p_alt]))

    return P


def syspermute(X, perm, dim):
    """
    Permutes order of subsystems in the multipartite operator X.

    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    subsystems.
    """

    # If p is defined using np.array(), then it must first be converted
    # to a numpy array, or else the reshaping below won't work.
    X = np.array(X)

    n = len(dim)
    d = X.shape

    perm = np.array(perm)
    dim = np.array(dim)

    if d[0] == 1 or d[1] == 1:
        # For a pure state
        perm = perm - 1
        tmp = np.reshape(X, dim)
        q = np.reshape(np.transpose(tmp, perm), d)

        return q
    elif d[0] == d[1]:
        # For a mixed state (density matrix)
        perm = perm - 1
        perm = np.append(perm, n + perm)
        dim = np.append(dim, dim)
        tmp = np.reshape(X, dim)
        Y = np.reshape(np.transpose(tmp, perm), d)

        return Y


def tensor(*args):
    """
    Takes the tensor product of an arbitrary number of matrices/vectors.
    """

    M = 1

    for j in range(len(args)):
        if isinstance(args[j], list):
            for k in range(args[j][1]):
                M = np.kron(M, args[j][0])
        else:
            M = np.kron(M, args[j])

    return M


def Tr(A):
    """
    Takes the trace of the matrix A, which should be specified as a numpy 2d array.
    """

    return np.trace(A)


def trace_distance_pure_states(psi, phi):
    """
    Computes the squared trace distance between two pure states psi and phi,
    i.e.,

    || |psi><psi|-|phi><phi| ||_1^2

    """

    if psi.shape[1] == 1:  # If psi is specified as a state vector
        psi = psi @ dag(psi)
    if phi.shape[1] == 1:  # If phi is specified as a state vector
        phi = phi @ dag(phi)

    return 1 - Tr(psi @ phi)


def trace_norm(X):
    """
    Finds the trace norm of the matrix X. (Sum of the singular values.)
    """

    return norm(X, ord="nuc")


def unitary_distance(U, V):
    """
    Checks whether two unitaries U and V are the same (taking into account global phase) by using the distance measure:

    1-(1/d)*|Tr[UV^†]|,

    where d is the dimension of the space on which the unitaries act.

    U and V are the same if and only if this is equal to zero; otherwise, it is greater than zero.
    """

    d = U.shape[0]

    return 1 - (1 / d) * np.abs(Tr(U @ dag(V)))
