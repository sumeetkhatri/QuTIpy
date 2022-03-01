'''
This code is part of QuTIpy.

(c) Copyright Sumeet Khatri, 2022

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
'''

import numpy as np

from qutipy.general_functions import Tr
from qutipy.fermions import jordan_wigner



def cov_matrix_fermi(X,n,rep='JW'):

    '''
    Generates the covariance matrix associated with the operator X. The underlying
    calculations are done using the specified representation, although the matrix
    itself is independent of the representation used for the calculation.
    '''

    G=np.zeros((2*n,2*n),dtype=complex)

    _,c=jordan_wigner(n)

    for j in range(1,2*n+1):
        for k in range(1,2*n+1):
            G[j-1,k-1]=(1j/2)*Tr(X@(c[j]@c[k]-c[k]@c[j]))

    return G