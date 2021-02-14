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

from qutipy.general_functions import eye
from qutipy.channels import generate_channel_isometry




def Pauli_channel(px,py,pz):

    '''
    Generates the Kraus operators, an isometric extension, and a unitary
    extension of the one-qubit Pauli channel specified by the parameters px, py, pz.
    '''

    pI=1-px-py-pz

    Sx=np.matrix([[0,1],[1,0]])
    Sy=np.matrix([[0,-1j],[1j,0]])
    Sz=np.matrix([[1,0],[0,-1]])

    K=[np.sqrt(pI)*eye(2),np.sqrt(px)*Sx,np.sqrt(py)*Sy,np.sqrt(pz)*Sz]

    V,U=generate_channel_isometry(K,2,2)

    return K,V,U