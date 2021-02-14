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


def Choi_to_Natural(C_AB,dimA,dimB):

    '''
    Takes the Choi representation of a map and outputs its natural representation.

    The Choi represenatation Q of the channel acts as:
    
        vec(N(rho))=Q*vec(rho),
    
    where N is the channel in question. It can be obtained from the Choi representation
    with a simple reshuffling of indices.
    '''

    C_AB=np.array(C_AB)

    return np.matrix(np.reshape(C_AB,[dimA,dimB,dimA,dimB]).transpose((0,2,1,3)).reshape([dimA*dimA,dimB*dimB])).T
