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

from qutipy.su import su_generators
from qutipy.general_functions import eye


def state_from_coherence_vector(n,d,state=True):

    '''
    Uses the supplied coherence vector n to generate the corresponding state via

        rho=(1/d)*(eye(d)+n*L),

    where L are the su(d) generators.

    If state=True, then n is a vector of length d^2-1. Otherwise, if state=False, then n is a vector of length d^2
    '''

    L=su_generators(d)

    if state:
        rho=np.matrix((1/d)*eye(d),dtype=np.complex128)
        for i in range(1,len(L)):
            rho+=(1/d)*n[i-1]*L[i]
        return rho
    
    else:
        X=np.matrix(np.zeros((d,d)),dtype=np.complex128)
        for i in range(len(L)):
            X+=(1/d)*n[i]*L[i]
        return X