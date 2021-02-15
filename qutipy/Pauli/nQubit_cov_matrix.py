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

from qutipy.general_functions import Tr
from qutipy.Pauli import nQubit_quadratures



def nQubit_cov_matrix(X,n):

    '''
    Using the n-qubit quadrature operators, we define the n-qubit "covariance matrix"
    as follows:

    V_{i,j}=Tr[X*S_i*S_j]
    '''


    S=nQubit_quadratures(n)

    V=np.array(np.zeros((2*n,2*n)),dtype=np.complex128)

    for i in range(2*n):
        for j in range(2*n):
            V[i,j]=Tr(X@S[i+1]@S[j+1])
    
    return V