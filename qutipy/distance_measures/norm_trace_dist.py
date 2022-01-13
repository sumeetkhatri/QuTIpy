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
    where rho and sigma are quantum states. More generally, they can be Hermitian
    operators.
    '''

    if sdp:
        if not dual:

            dim=rho.shape[0]

            L1=cvx.Variable((dim,dim),hermitian=True)
            L2=cvx.Variable((dim,dim),hermitian=True)

            c=[L1>>0,L2>>0,eye(dim)-L1>>0,eye(dim)-L2>>0]

            obj=cvx.Maximize(cvx.real(cvx.trace((L1-L2)@(rho-sigma))))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-7)

            return (1/2)*prob.value

        elif dual:

            dim=rho.shape[0]

            Y1=cvx.Variable((dim,dim),hermitian=True)
            Y2=cvx.Variable((dim,dim),hermitian=True)
            
            c=[Y1>>0,Y2>>0,Y1>>rho-sigma,Y2>>-(rho-sigma)]

            obj=cvx.Minimize(cvx.real(cvx.trace(Y1+Y2)))

            prob=cvx.Problem(obj,c)
            prob.solve(verbose=display,eps=1e-7)

            return (1/2)*prob.value
    else:
        return (1/2)*trace_norm(rho-sigma)