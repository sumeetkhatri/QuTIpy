'''
This code is part of QuTIpy.

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

from qutipy.general_functions import dag,tensor,eye
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy


def apply_channel(K,rho,sys=None,dim=None,adjoint=False):

    '''
    Applies the channel with Kraus operators in K to the state rho on
    systems specified by sys. The dimensions of the subsystems on which rho
    acts are given by dim.

    If adjoint is True, then this function applies the adjoint of the given
    channel.
    '''

    if isinstance(rho,cvxpy.Variable):
        rho=cvxpy_to_numpy(rho)
        rho_out=apply_channel(K,rho,sys,dim,adjoint)
        return numpy_to_cvxpy(rho_out)

    if adjoint==True:
        K_tmp=K
        K=[]
        K=[dag(K_tmp[i]) for i in range(len(K_tmp))]

    if sys==None:
        return np.sum([K[i]@rho@dag(K[i]) for i in range(len(K))],0)
    else:
        A=[]
        for i in range(len(K)):
            X=1
            for j in range(len(dim)):
                if j+1==sys:
                    X=tensor(X,K[i])
                else:
                    X=tensor(X,eye(dim[j]))
            A.append(X)

        return np.sum([A[i]@rho@dag(A[i]) for i in range(len(A))],0)