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


def MaxEnt_state(dim,normalized=True,density_matrix=True):

    '''
    Generates the dim-dimensional maximally entangled state, which is defined as

    (1/sqrt(dim))*(|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If normalized=False, then the function returns the unnormalized maximally entangled
    vector.

    If density_matrix=True, then the function returns the state as a density matrix.
    '''

    if normalized:
        Bell=(1./np.sqrt(dim))*np.matrix(np.sum([ket(dim,[i,i]) for i in range(dim)],0))
        if density_matrix:
            return Bell*Bell.H
        else:
            return Bell
    else:
        Gamma=np.matrix(np.sum([ket(dim,[i,i]) for i in range(dim)],0))
        if density_matrix:
            return Gamma*Gamma.H
        else:
            return Gamma