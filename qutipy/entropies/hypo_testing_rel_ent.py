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
import cvxpy as cvx

from qutipy.general_functions import eye



def hypo_testing_rel_ent(rho,sigma,eps,dual=False,log=False,display=False,return_all=False):

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

        prob.solve(verbose=display,eps=1e-7)

        if not log:
            if return_all:
                return prob.value,L.value
            else:
                return prob.value
        else:
            if return_all:
                return -np.log2(prob.value),L.value
            else:
                return -np.log2(prob.value)

    elif dual:

        dim=rho.shape[0]

        Z=cvx.Variable((dim,dim),hermitian=True)
        mu=cvx.Variable()

        c=[mu>=0,mu*rho<<sigma+Z,Z>>0]

        obj=cvx.Maximize(cvx.real(mu*(1-eps)-cvx.trace(Z)))

        prob=cvx.Problem(obj,c)
        prob.solve(verbose=display,eps=1e-7)

        if not log:
            if return_all:
                return prob.value,Z.value,mu.value
            else:
                return prob.value
        else:
            if return_all:
                return -np.log2(prob.value),Z.value,mu.value
            else:
                return -np.log2(prob.value)