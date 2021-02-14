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


from qutipy.channels import Pauli_channel



def depolarizing_channel(p):

    '''
    For 0<=p<=1, this returns the one-qubit Pauli channel given by px=py=pz=p/3.
    '''

    return Pauli_channel(p/3.,p/3.,p/3.)