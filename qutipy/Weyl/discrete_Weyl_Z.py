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

from qutipy.general_functions import ket


def discrete_Weyl_Z(d):

    '''
    Generates the Z phase operators.
    '''

    w=np.exp(2*np.pi*1j/d)

    Z=ket(d,0)*ket(d,0).H

    for i in range(1,d):
        Z=Z+w**i*ket(d,i)*ket(d,i).H

    return Z