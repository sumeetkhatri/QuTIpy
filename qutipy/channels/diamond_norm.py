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

from qutipy.general_functions import syspermute,ket,eye


def diamond_norm(J,dimA,dimB,display=True):

    '''
    Computes the diamond norm of a superoperator with Choi representation J.
    dimA is the dimension of the input space of the channel, and dimB is the
    dimension of the output space.

    The form of the SDP used comes from Theorem 3.1 of:
        
        'Simpler semidefinite programs for completely bounded norms',
            Chicago Journal of Theoretical Computer Science 2013,
            by John Watrous
    '''

    '''
    The Choi representation J in the above paper is defined using a different
    convention:
        J=(N\otimes I)(|Phi^+><Phi^+|).
    In other words, the channel N acts on the first half of the maximally-
    entangled state, while the convention used throughout this code stack
    is
        J=(I\otimes N)(|Phi^+><Phi^+|).
    We thus use syspermute to convert to the form used in the aforementioned
    paper.
    '''

    J=np.matrix(syspermute(J,[2,1],[dimA,dimB]))

    X=cvx.Variable((dimA*dimB,dimA*dimB))
    rho0=cvx.Variable((dimA,dimA),PSD=True)
    rho1=cvx.Variable((dimA,dimA),PSD=True)

    M=cvx.kron(ket(2,0)*ket(2,0).H,cvx.kron(eye(dimB),rho0))+cvx.kron(ket(2,0)*ket(2,1).H,X)+cvx.kron(ket(2,1)*ket(2,0).H,X.H)+cvx.kron(ket(2,1)*ket(2,1).H,cvx.kron(eye(dimB),rho1))

    c=[]
    c+=[M>>0,cvx.trace(rho0)==1,cvx.trace(rho1)==1]

    obj=cvx.Maximize((1./2.)*cvx.real(cvx.trace(J.H*X))+(1./2.)*cvx.real(cvx.trace(J*X.H)))

    prob=cvx.Problem(obj,constraints=c)

    prob.solve(solver=cvx.CVXOPT,verbose=display,eps=1e-8)
    #prob.solve(verbose=display)

    return prob.value