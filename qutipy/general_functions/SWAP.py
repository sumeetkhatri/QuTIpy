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

from qutipy.general_functions import tensor,ket,eye,syspermute,partial_transpose


def SWAP(sys,dim):

    '''
    Generates a swap matrix between the pair of systems in sys. dim is a list
    of the dimensions of the subsystems.

    For example, SWAP([1,2],[2,2]) generates the two-qubit swap matrix.
    '''

    dim_total=np.product(dim)

    n=len(dim)
    sys_rest=list(np.setdiff1d(range(1,n+1),sys))
    perm=sys+sys_rest
    p={}

    for i in range(1,n+1):
        p[i]=perm[i-1]

    p2={v:k for k,v in p.items()}

    perm_rearrange=list(p2.values())

    dim1=dim[sys[0]-1] # Dimension of the first subsystem to be swapped
    dim2=dim[sys[1]-1] # Dimension of the second subsystem to be swapped

    dim_rest=int(float(dim_total)/float(dim1*dim2))

    G1=np.matrix(np.sum([ket(dim1,[i,i]) for i in range(dim1)],0))
    G2=np.matrix(np.sum([ket(dim2,[i,i]) for i in range(dim2)],0))

    G=G1*G2.H

    S=partial_transpose(G,[2],[(dim1,dim2),(dim1,dim2)])

    P=tensor(S,eye(dim_rest))

    p_alt=list(np.array(list(p.values()))-1)

    P=syspermute(P,perm_rearrange,list(np.array(dim)[p_alt]))

    return P