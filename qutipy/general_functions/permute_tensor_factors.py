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

from qutipy.general_functions import generate_all_kets, syspermute,dag


def permute_tensor_factors(perm,dims):

    '''
    Generates the permutation operator that permutes the tensor factors according
    to the given permutation.
    
    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    tensor factors.
    '''

    K=generate_all_kets(dims)

    dim=np.prod(dims)

    W=np.zeros((dim,dim),dtype=complex)

    for ket in K:
        W=W+syspermute(ket,perm,dims)@dag(ket)

    return W