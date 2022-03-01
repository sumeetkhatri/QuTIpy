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
from scipy.linalg import expm

from qutipy.fermions import jordan_wigner
from qutipy.general_functions import eye,dag,ket,tensor


def coherent_state_fermi(A,rep='JW',density_matrix=False):

    '''
    Generates the fermionic coherent state vector for n modes, where A is a complex
    anti-symmetric n x n matrix. The matrix A should be at least 2 x 2 -- for one mode,
    the coherent state is the just the vacuum.

    The definition being used here comes from

        A. Perelomov. Generalized Coherent States and Their Applications (Sec. 9.2)

    '''

    n=np.shape(A)[0]  # Number of modes

    a,_=jordan_wigner(n)

    At=np.zeros((2**n,2**n),dtype=complex)

    N=np.linalg.det(eye(n)+A@dag(A))**(1/4)

    for i in range(1,n+1):
        for j in range(1,n+1):
            At=At+(-1/2)*A[i-1,j-1]*dag(a[j])@dag(a[i])

    vac=tensor([ket(2,0),n])

    if not density_matrix:
        return (1/N)*expm(At)@vac
    else:
        coh=(1/N)*expm(At)@vac
        return coh@dag(coh)
    
