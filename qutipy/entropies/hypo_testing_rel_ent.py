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
import cvxpy as cvx

from qutipy.general_functions import eye



def hypo_testing_rel_ent(rho,sigma,eps,dual=False,display=False):

    '''
    Calculates the eps-hypothesis testing relative entropy of the two states
    rho and sigma.
    '''

    if not dual:
        
        dim=rho.shape[0]

        L=cvx.Variable((dim,dim),hermitian=True)

        c=[L>>0,eye(dim)-L>>0]
        c+=[cvx.real(cvx.trace(L@rho))>=1-eps]

        obj=cvx.Minimize(cvx.real(cvx.trace(L@sigma)))
        prob=cvx.Problem(obj,constraints=c)

        prob.solve(verbose=display)

        return -np.log2(prob.value)

    elif dual:

        dim=rho.shape[0]

        Y=cvx.Variable((dim,dim),hermitian=True)
        l=cvx.Variable()

        c=[l>=0,Y>>sigma,Y-l*rho>>0]

        obj=cvx.Maximize(cvx.real(-cvx.trace(Y)+cvx.trace(sigma)+l*(1-eps)))

        prob=cvx.Problem(obj,c)
        prob.solve(verbose=display)

        return -np.log2(prob.value)