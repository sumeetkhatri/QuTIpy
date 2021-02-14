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


def partial_transpose(X,sys,dim):

    '''
    Takes the partial transpose on systems given by sys. dim is a list of 
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
    '''

    if X.shape[1]==1:
        X=X*X.H

    X=np.array(X)

    n=len(dim)  # Number of subsystems in the operator

    if type(dim[0])==tuple or type(dim[0])==list:  # When the operator is a non-square matrix
        dim_row=[dim[i][0] for i in range(n)]
        dim_col=[dim[i][1] for i in range(n)]
    elif type(dim[0])==int:  # When the operator is a square matrix
        dim_row=dim
        dim_col=dim

    X_reshape=np.reshape(X,dim_row+dim_col)

    axes=list(range(2*n))

    for i in range(len(sys)):
        axes[sys[i]-1],axes[n+sys[i]-1]=axes[n+sys[i]-1],axes[sys[i]-1]
        if type(dim[0])==tuple or type(dim[0])==list:
            dim[sys[i]-1]=list(dim[sys[i]-1])
            dim[sys[i]-1][0],dim[sys[i]-1][1]=dim[sys[i]-1][1],dim[sys[i]-1][0]
            #dim[sys[i]-1]=tuple(dim[sys[i]-1])
        else:
            continue

    X_reshape=np.transpose(X_reshape,tuple(axes))

    if type(dim[0])==tuple or type(dim[0])==list:
        dim_row=[dim[i][0] for i in range(n)]
        dim_col=[dim[i][1] for i in range(n)]
        dim_total=(np.product(dim_row),np.product(dim_col))
    elif type(dim[0])==int:
        dim_row=dim
        dim_col=dim
        dim_total=(np.product(dim),np.product(dim))

    X_new=np.reshape(X_reshape,dim_total)

    return np.matrix(X_new)