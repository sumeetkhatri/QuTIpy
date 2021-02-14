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


from qutipy.general_functions import eye,tensor


def generate_nQubit_Pauli(indices):

    '''
    Generates a tensor product of Pauli operators for n qubits. indices is a list
    of indices i specifying the Pauli operator for each tensor factor. i=0 is the identity, i=1 is sigma_x,
    i=2 is sigma_y, and i=3 is sigma_z.
    '''

    Id=eye(2)
    Sx=np.matrix([[0,1],[1,0]])
    Sy=np.matrix([[0,-1j],[1j,0]])
    Sz=np.matrix([[1,0],[0,-1]])

    out=1

    for index in indices:
        if index==0:
            out=tensor(out,Id)
        elif index==1:
            out=tensor(out,Sx)
        elif index==2:
            out=tensor(out,Sy)
        elif index==3:
            out=tensor(out,Sz)
    
    return out