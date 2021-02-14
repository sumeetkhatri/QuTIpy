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

from qutipy.general_functions import ket
from qutipy.Pauli import generate_nQubit_Pauli_X,generate_nQubit_Pauli_Z


def nQubit_quadratures(n):

    '''
    Returns the list of n-qubit "quadrature" operators, which are defined as
    (for two qubits)

        S[0]=Sx \otimes Id
        S[1]=Sz \otimes Id
        S[2]=Id \otimes Sx
        S[3]=Id \otimes Sz

    In general, for n qubits:

        S[0]=Sx \otimes Id \otimes ... \otimes Id
        S[1]=Sz \otimes Id \otimes ... \otimes Id
        S[2]=Id \otimes Sx \otimes ... \otimes Id
        S[3]=Id \otimes Sz \otimes ... \otimes Id
        .
        .
        .
        S[2n-2]=Id \otimes Id \otimes ... \otimes Sx
        S[2n-1]=Id\otimes Id \otimes ... \otimes Sz
    '''

    S={}

    #Sx=np.matrix([[0,1],[1,0]])
    #Sz=np.matrix([[1,0],[0,-1]])

    count=0

    for i in range(1,2*n+1,2):
        v=list(np.array(ket(n,count).H,dtype=np.int).flatten())
        S[i]=generate_nQubit_Pauli_X(v)
        S[i+1]=generate_nQubit_Pauli_Z(v)
        count+=1

    return S