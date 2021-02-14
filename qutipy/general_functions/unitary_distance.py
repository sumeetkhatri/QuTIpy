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

from qutipy.general_functions import Tr

def unitary_distance(U,V):
    
    '''
    Checks whether two unitaries U and V are the same (taking into account global phase) by using the distance measure:
    
    1-(1/d)*|Tr[UV^\dagger]|,
    
    where d is the dimension of the space on which the unitaries act.
    
    U and V are the same if and only if this is equal to zero; otherwise, it is greater than zero.
    '''
    
    U=np.matrix(U)
    V=np.matrix(V)

    d=U.shape[0]
    
    return 1-(1/d)*np.abs(Tr(U*V.H))