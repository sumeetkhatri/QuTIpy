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
from qutipy.general_functions import tensor


def Natural_representation(K):

    '''
    Calculates the natural representation of the channel (in the standard basis)
    given by the Kraus operators in K. In terms of the Kraus operators, the natural
    representation of the channel in the standard basis is given by

    N=sum_i K_i ⊗ conj(C_i),

    where the sum is over the Kraus operators K_i in K.
    '''

    return np.matrix(np.sum([tensor(k,np.conjugate(k)) for k in K],1))