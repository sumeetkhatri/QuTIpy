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
from qutipy.general_functions import ket
from qutipy.Weyl import generate_nQudit_X, generate_nQudit_Z


def nQudit_quadratures(d,n):

    '''
    Returns the list of n-qudit "quadrature" operators, which are defined as
    (for two qudits)

        S[0]=X(0) ⊗ Id
        S[1]=Z(0) ⊗ Id
        S[2]=Id ⊗ X(0)
        S[3]=Id ⊗ Z(0)

    In general, for n qubits:

        S[0]=X(0) ⊗ Id ⊗ ... ⊗ Id
        S[1]=Z(0) ⊗ Id ⊗ ... ⊗ Id
        S[2]=Id ⊗ X(0) ⊗ ... ⊗ Id
        S[3]=Id ⊗ Z(0) ⊗ ... ⊗ Id
        .
        .
        .
        S[2n-2]=Id ⊗ Id ⊗ ... ⊗ X(0)
        S[2n-1]=Id ⊗ Id ⊗ ... ⊗ Z(0)
    '''

    S={}

    count=0

    for i in range(1,2*n+1,2):
        v=list(np.array(ket(n,count).H,dtype=np.int).flatten())
        S[i]=generate_nQudit_X(d,v)
        S[i+1]=generate_nQudit_Z(d,v)
        count+=1

    return S