'''
This code is part of QuTIPy.

(c) Copyright Sumeet Khatri, 2021

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
'''


import numpy as np
import cvxpy

from qutipy.general_functions import syspermute,Tr
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy


def partial_trace(X,sys,dim):

    '''
    sys is a list of systems over which to take the partial trace (i.e., the
    systems to discard).

    Example: If rho_AB is a bipartite state with dimA the dimension of system A 
    and dimB the dimension of system B, then

    partial_trace(rho_AB,[2],[dimA,dimB]) gives the density matrix on

    system A, i.e., rho_A:=TrX[rho_AB].

    Similarly, partial_trace(rho_AB,[1],[dimA,dimB]) discards the first subsystem,
    returning the density matrix of system B.

    If rho_ABC is a tripartite state, then, e.g.,

    partial_trace(rho_ABC,[1,3],[dimA,dimB,dimC])

    discards the first and third subsystems, so that we obtain the density
    matrix for system B.

    '''

    if isinstance(X,cvxpy.Variable):
        X=cvxpy_to_numpy(X)
        X_out=partial_trace(X,sys,dim)
        return numpy_to_cvxpy(X_out)


    if not sys:  # If sys is empty, just return the original operator
        return X
    elif len(sys)==len(dim):  # If tracing over all systems
        return Tr(X)
    else:

        if X.shape[1]==1:
            X=X*X.H

        num_sys=len(dim)
        total_sys=range(1,num_sys+1)

        dims_sys=[dim[s-1] for s in sys] # Dimensions of the system to be traced over
        dims_keep=[dim[s-1] for s in list(set(total_sys)-set(sys))]
        dim_sys=np.product(dims_sys)
        dim_keep=np.product(dims_keep)

        perm=sys+list(set(total_sys)-set(sys))
        X=syspermute(X,perm,dim)

        X=np.array(X)
        dim=[dim_sys]+dims_keep
        X_reshape=np.reshape(X,dim+dim)
        X_reshape=np.sum(np.diagonal(X_reshape,axis1=0,axis2=len(dim)),axis=-1)
        X=np.reshape(X_reshape,(dim_keep,dim_keep))

        return np.matrix(X)