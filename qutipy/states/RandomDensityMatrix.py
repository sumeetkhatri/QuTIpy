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

from qutipy.general_functions import dag,Tr


def RandomDensityMatrix(dim,*args):

    '''
    Generates a random density matrix.
    
    Optional argument is for the rank r of the state.
    
    Optional argument comp is for whether the state should have
    complex entries
    '''

    args=np.array(args)

    if args.size==0:
        r=dim
    else:
        r=args[0]

    gin=np.random.randn(dim,r)+1j*np.random.randn(dim,r)
    rho=gin@dag(gin)
    
    return rho/Tr(rho)