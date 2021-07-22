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

from qutipy.general_functions import eye,trace_norm



def norm_trace_dist(rho,sigma,sdp=False,dual=False,display=False):

    '''
    Calculates the normalized trace distance (1/2)*||rho-sigma||_1 using an SDP,
    where rho and sigma are quantum states.
    '''

    if sdp:
        if not dual:

            dim=rho.shape[0]

            L=cvx.Variable((dim,dim),hermitian=True)

            c=[L>>0,eye(dim)-L>>0]

            obj=cvx.Maximize(cvx.real(cvx.trace(L@(rho-sigma))))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display)

            return prob.value

        elif dual:

            dim=rho.shape[0]

            Z=cvx.Variable((dim,dim),hermitian=True)
            
            c=[Z>>0,Z>>rho-sigma]

            obj=cvx.Minimize(cvx.real(cvx.trace(Z)))

            prob=cvx.Problem(obj,c)
            prob.solve(verbose=display)

            return prob.value
    else:
        return (1/2)*trace_norm(rho-sigma)