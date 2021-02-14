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


def GHZ_state(dim,n,density_matrix=True):

    # Last modified: 15 June 2020

    '''
    Generates the n-party GHZ state in dim-dimensions for each party, which is defined as

        |GHZ_n> = (1/sqrt(dim))*(|0,0,...,0> + |1,1,...,1> + ... + |d-1,d-1,...,d-1>)

    If density_matrix=True, then the function returns the state as a density matrix.
    '''

    GHZ=(1/np.sqrt(dim))*np.matrix(np.sum([ket(dim,[i]*n) for i in range(dim)],0))

    if density_matrix:
        return GHZ*GHZ.H
    else:
        return GHZ