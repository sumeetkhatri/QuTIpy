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

from qutipy.Weyl import nQudit_quadratures
from qutipy.general_functions import Tr


def nQudit_cov_matrix(X,d,n):

    '''
    Generates the matrix of second moments (aka covariance matrix) of an
    n-qudit operator X.
    '''


    S=nQudit_quadratures(d,n)

    V=np.matrix(np.zeros((2*n,2*n)),dtype=np.complex128)

    for i in range(2*n):
        for j in range(2*n):
            #V[i,j]=np.trace(X*(S[i+1]*S[j+1].H+S[j+1].H*S[i+1]))
            V[i,j]=Tr(X*S[i+1]*S[j+1].H)  # Use this instead to be consistent with the qubit.

    return V