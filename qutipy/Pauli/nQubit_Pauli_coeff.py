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


import itertools
import numpy as np

from qutipy.Pauli import generate_nQubit_Pauli
from qutipy.general_functions import dag,Tr


def nQubit_Pauli_coeff(X,n):

    '''
    Generates the coefficients of the matrix X in the n-qubit Pauli basis.
    The coefficients c_{alpha} are such that

    X=(1/2^n)\sum_{alpha} c_alpha \sigma_alpha

    The coefficients are returned in lexicographical ordering.
    '''

    indices=list(itertools.product(*[range(0,4)]*n))

    C=[]

    for index in indices:
        sigma_i=generate_nQubit_Pauli(index)
        C.append(Tr(dag(sigma_i)@X))

    return C