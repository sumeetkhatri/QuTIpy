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


def syspermute(X,perm,dim):

    '''
    Permutes order of subsystems in the multipartite operator X.
    
    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    subsystems.
    '''

    # If p is defined using np.matrix(), then it must first be converted
    # to a numpy array, or else the reshaping below won't work.
    X=np.array(X) 


    n=len(dim)
    d=X.shape

    perm=np.array(perm)
    dim=np.array(dim)

    if d[0]==1 or d[1]==1:
        # For a pure state
        perm=perm-1
        tmp=np.reshape(X,dim)
        q=np.reshape(np.transpose(tmp,perm),d)

        return q
    elif d[0]==d[1]:
        # For a mixed state (density matrix)
        perm=perm-1
        perm=np.append(perm,n+perm)
        dim=np.append(dim,dim)
        tmp=np.reshape(X,dim)
        Y=np.reshape(np.transpose(tmp,perm),d)

        return Y