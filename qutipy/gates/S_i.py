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

from qutipy.general_functions import tensor,syspermute,eye


def S_i(i,n):

    '''
    Generates the matrix for the S gate applied to the ith qubit.
    n is the total number of qubits. The S gate is defined as:

        S:=[[1 0],
            [0 1j]]

    It is one of the generators of the Clifford group.
    '''

    dims=2*np.ones(n)
    dims=dims.astype(int)
    indices=np.linspace(1,n,n)
    indices_diff=np.setdiff1d(indices,i)
    perm_arrange=np.append(np.array([i]),indices_diff)
    perm_rearrange=np.zeros(n)

    for i in range(n):
        perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1

    perm_rearrange=perm_rearrange.astype(int)
    S=np.matrix([[1,0],[0,1j]])
    out_temp=tensor(S,[eye(2),n-1])
    out=syspermute(out_temp,perm_rearrange,dims)

    return np.matrix(out)