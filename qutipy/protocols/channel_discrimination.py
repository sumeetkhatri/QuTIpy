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

from qutipy.general_functions import trace_norm,eye,partial_trace
from qutipy.general_functions.syspermute import syspermute
from qutipy.misc import cvxpy_to_numpy,numpy_to_cvxpy
from qutipy.channels import diamond_norm


def channel_discrimination(J0,J1,dimA,dimB,p,succ=False,sdp=False,dual=False,display=False):

    '''
    Calculates the optimal error probability for quantum channel discrimination, with prior
    probability p for the channel with Choi representation J1.

    J0 and J1 are the Choi representations of the two channels. dimA and dimB are the input
    and output dimensions, respectively, of the channels.

    If succ=True, then this function returns the optimal success probability instead.
    If sdp=True, then this function calculates the optimal value (error or success 
    probability) using an SDP.
    '''


    if sdp:

        if not dual:

            # Need the following syspermute because the cvxpy kron function
            # requires a constant in the first argument
            J0=syspermute(J0,[2,1],[dimA,dimB])
            J1=syspermute(J1,[2,1],[dimA,dimB])

            Q0=cvx.Variable((dimA*dimB,dimA*dimB),hermitian=True)
            Q1=cvx.Variable((dimA*dimB,dimA*dimB),hermitian=True)
            rho=cvx.Variable((dimA,dimA),hermitian=True)

            c=[Q0>>0,Q1>>0,rho>>0,cvx.real(cvx.trace(rho))==1,Q0+Q1==cvx.kron(eye(dimB),rho)]

            obj=cvx.Minimize(cvx.real(p*cvx.trace(Q1@J0)+(1-p)*cvx.trace(Q0@J1)))
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-7)

            p_err=prob.value

            if succ:
                return 1-p_err
            else:
                return p_err


        elif dual:

            mu=cvx.Variable()
            W=cvx.Variable((dimA*dimB,dimA*dimB),hermitian=True)

            WA=numpy_to_cvxpy(partial_trace(cvxpy_to_numpy(W),[2],[dimA,dimB]))

            c=[W<<p*J0,W<<(1-p)*J1,mu*eye(dimA)<<WA]

            obj=cvx.Maximize(mu)
            prob=cvx.Problem(obj,constraints=c)

            prob.solve(verbose=display,eps=1e-7)

            p_err=prob.value

            if succ:
                return 1-p_err
            else:
                return p_err

    else:
        p_err=(1/2)*(1-diamond_norm(p*J0-(1-p)*J1,dimA,dimB,display=display))
        if succ:
            return 1-p_err
        else:
            return p_err
