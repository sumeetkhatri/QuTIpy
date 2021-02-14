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


import cvxpy as cvx
import numpy as np

from qutipy.general_functions import partial_trace,eye


def check_kext(rhoAB,dimA,dimB,k,display=False):

    '''
    Checks if the bipartite state rhoAB is k-extendible.
    '''

    all_sys=list(range(1,k+2))
    dim=[dimA]+[dimB]*k

    t=cvx.Variable()
    R=cvx.Variable((dimA*dimB**k,dimA*dimB**k),hermitian=True)

    obj=cvx.Maximize(t)

    c=[R-t*eye(dimA*dimB**k)>>0]

    for j in range(2,k+2):

        sys=list(np.setdiff1d(all_sys,[1,j]))

        R_ABj=partial_trace(R,sys,dim)

        c.append(R_ABj==rhoAB)

    prob=cvx.Problem(obj,constraints=c)

    prob.solve(verbose=display)

    return prob.value,R.value
