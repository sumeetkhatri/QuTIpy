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


def nQubit_mean_vector(X,n):

    '''
    Using the n-qubit quadrature operators, we define the n-qubit "mean vector" as
    follows:

        r_i=Tr[X*S_i]
    '''


    S=nQubit_quadratures(n)

    r=np.matrix(np.zeros((2*n,1)),dtype=np.complex128)
    #r=np.matrix(np.zeros((2*n,1)),dtype=object)

    for i in range(2*n):
        r[i,0]=Tr(X*S[i+1])

    return r