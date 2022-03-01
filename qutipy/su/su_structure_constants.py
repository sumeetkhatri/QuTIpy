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

from qutipy.su import su_generators
from qutipy.general_functions import Tr


def su_structure_constants(d):

    '''
    Generates the structure constants corresponding to the su(d)
    basis elements. They are defined as follows:

        f_{i,j,k}=(1/(1j*d^2))*Tr[S_k*[S_i,S_j]]

        g_{i,j,k}=(1/d^2)*Tr[S_k*{S_i,S_j}]
    
    '''

    f={}
    g={}

    S=su_generators(d)

    for i in range(1,d**2):
        for j in range(1,d**2):
            for k in range(1,d**2):

                f[(i,j,k)]=(1/(1j*d**2))*Tr(S[k]@(S[i]@S[j]-S[j]@S[i]))

                g[(i,j,k)]=(1/d**2)*Tr(S[k]@(S[i]@S[j]+S[j]@S[i]))

    return f,g