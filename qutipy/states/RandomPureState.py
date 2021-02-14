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
from numpy.linalg import norm

from qutipy.states import MaxEnt_state
from qutipy.general_functions import syspermute,tensor,eye


def RandomPureState(dim,rank=None):

    '''
    Generates a random pure state.

    For multipartite states, dim should be a list of dimensions for each
    subsystem. In this case, the rank variable is for the Schmidt rank. To specify
    the Schmidt rank, there has to be a bipartition of the systems, so that dim
    has only two elements.
    '''
    
    if rank==None:
        if type(dim)==list:
            dim=np.prod(dim)

        # Generate the real and imaginary parts of the components using numbers
        # sampled from the standard normal distribution (normal distribution with
        # mean zero and variance 1).
        psi=np.matrix(np.random.randn(dim)).H+1j*np.matrix(np.random.randn(dim)).H

        psi=psi/norm(psi)

        return psi
    else:
        dimA=dim[0]
        dimB=dim[1]
        k=rank
        psi_k=MaxEnt_state(k)
        a=np.matrix(np.random.rand(dimA*k)).H+1j*np.matrix(np.random.rand(dimA*k)).H
        b=np.matrix(np.random.rand(dimB*k)).H+1j*np.matrix(np.random.rand(dimB*k)).H

        psi_init=syspermute(tensor(a,b),[1,3,2,4],[k,dimA,k,dimB])

        psi=tensor(psi_k.H,eye(dimA*dimB))*psi_init

        psi=psi/norm(psi)

        return psi