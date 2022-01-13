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

from qutipy.general_functions import dag,tensor,ket
from qutipy.linalg import gram_schmidt
from qutipy.states import RandomStateVector


def generate_channel_isometry(K,dimA,dimB):

    '''
    Generates an isometric extension of the
    channel specified by the Kraus operators K. dimA is the dimension of the
    input space of the channel, and dimB is the dimension of the output space
    of the channel. If dimA=dimB, then the function also outputs a unitary
    extension of the channel given by a particular construction.
    '''

    dimE=len(K)

    V=np.sum([tensor(K[i],ket(dimE,i)) for i in range(dimE)],0)

    if dimA==dimB:
        # In this case, the unitary we generate has dimensions dimA*dimE x dimA*dimE
        U=tensor(V,dag(ket(dimE,0)))
        states=[V@ket(dimA,i) for i in range(dimA)]
        for i in range(dimA*dimE-dimA):
            states.append(RandomStateVector(dimA*dimE))

        states_new=gram_schmidt(states,dimA*dimE)

        count=dimA
        for i in range(dimA):
            for j in range(1,dimE):
                U=U+tensor(states_new[count],dag(ket(dimA,i)),dag(ket(dimE,j)))
                count+=1
        
        return V,np.array(U)
    else:
        return V