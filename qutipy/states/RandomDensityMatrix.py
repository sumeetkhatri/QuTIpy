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


def RandomDensityMatrix(dim,comp=True,*args):

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

    if comp:
        gin=np.matrix(np.random.randn(dim,r)+1j*np.random.randn(dim,r))
        rho=gin*gin.H
    else:
        gin=np.matrix(np.random.randn(dim,r))
        rho=gin*gin.H

    return rho/np.trace(rho)