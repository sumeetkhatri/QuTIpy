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


from cvxpy.settings import CVXOPT
import numpy as np
import cvxpy as cvx

from qutipy.general_functions import trace_norm,eye


def state_discrimination(rho,sigma,p,succ=False,sdp=False,dual=False,display=False):

    '''
    Calculates the optimal error probability for quantum state discrimination, with prior
    probability p for the state rho.

    If succ=True, then this function returns the optimal success probability instead.
    If sdp=True, then this function calculates the optimal value (error or success 
    probability) using an SDP.
    '''

    if sdp:

        if not dual:

            dim=rho.shape[0]

            M=cvx.Variable((dim,dim),hermitian=True)

            c=[M>>0,eye(dim)-M>>0]

            obj=cvx.Minimize(cvx.real(p*cvx.trace((eye(dim)-M)@rho)+(1-p)*cvx.trace(M@sigma)))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-7)

            p_err=prob.value

            if succ:
                return 1-p_err
            else:
                return p_err


        elif dual:

            dim=rho.shape[0]

            W=cvx.Variable((dim,dim),hermitian=True)

            c=[W<<p*rho,W<<(1-p)*sigma]

            obj=cvx.Maximize(cvx.real(cvx.trace(W)))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-7)

            p_err=prob.value

            if succ:
                return 1-p_err
            else:
                return p_err

    else:
        p_err=(1/2)*(1-trace_norm(p*rho-(1-p)*sigma))
        if succ:
            return 1-p_err
        else:
            return p_err
