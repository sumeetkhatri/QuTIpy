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

from qutipy.general_functions import dag,ket,tensor
from qutipy.su import su_generators


def jordan_wigner(n):

    '''
    Generates the Jordan-Wigner representation of the fermionic creation, annihilation,
    and Majorana operators for an n-mode system.

    The convention for the Majorana operators is as follows:

        c_j=aj^{dag}+aj
        c_{n+j}=i(aj^{dag}-aj)

    '''

    s=ket(2,0)@dag(ket(2,1))

    S=su_generators(2)

    a={}  # Dictionary for the annihilation operators
    c={}  # Dictionary for the Majorana operators

    for j in range(1,n+1):
        a[j]=tensor([S[3],j-1],s,[S[0],n-j])
        c[j]=dag(a[j])+a[j]
        c[n+j]=1j*(dag(a[j])-a[j])

    return a,c