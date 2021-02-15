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

from qutipy.general_functions import dag,ket,eye


def su_generators(d):

    '''
    Generates the basis (aka generators) of the Lie algebra su(d)
    corresponding to the Lie group SU(d). The basis has d^2-1 elements.

    All of the generators are traceless and Hermitian. After adding the
    identity matrix, they form an orthogonal basis for all dxd matrices.

    The orthogonality condition is

        Tr[l_i*l_j]=d*delta_{i,j}

    (This is a particular convention we use here; there are other conventions.)

    For d=2, we get the Pauli matrices.
    '''

    L=[]

    L.append(eye(d))

    for l in range(d):
        for k in range(l):
            L.append(np.sqrt(d/2)*(ket(d,k)@dag(ket(d,l))+ket(d,l)@dag(ket(d,k))))
            L.append(np.sqrt(d/2)*(-1j*ket(d,k)@dag(ket(d,l))+1j*ket(d,l)@dag(ket(d,k))))

    for k in range(1,d):
        X=0
        for j in range(k):
            X+=ket(d,j)@dag(ket(d,j))
        
        L.append(np.sqrt(d/(k*(k+1)))*(X-k*ket(d,k)@dag(ket(d,k))))

    
    return L