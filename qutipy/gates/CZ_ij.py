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


from qutipy.general_functions import ket,tensor,syspermute,eye


def CZ_ij(i,j,n):

    '''
    CZ gate on qubits i and j, i being the control and j being the target.
    The total number of qubits is n. (Note that for the CZ gate it does matter
    which qubit is the control and which qubit is the target.)
    '''

    dims=2*np.ones(n)
    dims=dims.astype(int)

    indices=np.linspace(1,n,n)
    indices_diff=np.setdiff1d(indices,[i,j])

    perm_arrange=np.append(np.array([i,j]),indices_diff)
    perm_rearrange=np.zeros(n)

    for i in range(n):
        perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1

    perm_rearrange=perm_rearrange.astype(int)

    Sz=np.matrix([[1,0],[0,-1]])
    CZ=tensor(ket(2,0)*np.transpose(ket(2,0)),eye(2))+tensor(ket(2,1)*np.transpose(ket(2,1)),Sz)

    out_temp=tensor(CZ,[eye(2),n-2])

    out=syspermute(out_temp,perm_rearrange,dims)

    return out