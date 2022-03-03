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

import itertools
from qutipy.general_functions import ket


def generate_all_kets(dims):

    '''
    Generates the tensor-product orthonormal basis corresponding to vector spaces
    with dimensions in the list dims.

    ------------------------
    Example:

        generate_all_kets([2,3]) returns a list containing

            |0,0>
            |0,1>
            |0,2>
            |1,0>
            |1,1>
            |1,2>

    '''

    dims_set=[range(d) for d in dims]

    L=list(itertools.product(*dims_set))

    K=[]

    for l in L:
        K.append(ket(dims,l))

    return K
    