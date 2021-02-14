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


from qutipy.channels import Pauli_channel_nQubit


def depolarizing_channel_nQubits(n,p):

    '''
    For 0<=p<=1, this returns the n-qubit Pauli channel given by
    p[0]=1-p, p[i]=p/(2^(2*n)-1) for all i>=1.
    '''

    p=[1-p]+[p/(2**(2*n)-1) for i in range(2**(2*n)-1)]

    return Pauli_channel_nQubit(n,p,alt_repr=True)