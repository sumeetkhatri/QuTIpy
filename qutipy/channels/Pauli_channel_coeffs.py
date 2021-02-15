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
import itertools

from qutipy.Pauli import generate_nQubit_Pauli_X, generate_nQubit_Pauli_Z
from qutipy.general_functions import dag,Tr
from qutipy.channels import apply_channel



def Pauli_channel_coeffs(K,n,as_dict=False):

    '''
    Generates the coefficients c_{a,b} such that

        P(X^aZ^b)=c_{a,b}X^aZ^b,
    
    for the channel P with the Kraus operators in K.
    '''

    if as_dict:
        c={}
    else:
        c=[]

    S=list(itertools.product(*[range(0,2)]*n))
    #print(S)

    for a in S:
        for b in S:
            Xa=generate_nQubit_Pauli_X(list(a))
            Zb=generate_nQubit_Pauli_Z(list(b))
            if as_dict:
                c[(a,b)]=(1/2**n)*Tr(dag(Xa@Zb)@apply_channel(K,Xa@Zb))
            else:
                c.append((1/2**n)*Tr(dag(Xa@Zb)@apply_channel(K,Xa@Zb)))

    return c