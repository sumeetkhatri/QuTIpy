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
import itertools


from qutipy.Weyl import generate_nQudit_X, generate_nQudit_Z
from qutipy.general_functions import Tr


def nQudit_Weyl_coeff(X,d,n):

    '''
    Generates the coefficients of the operator X acting on n qudit
    systems.
    '''

    C={}

    S=list(itertools.product(*[range(0,d)]*n))

    for s in S:
        s=list(s)
        for t in S:
            t=list(t)
            G=generate_nQudit_X(d,s)*generate_nQudit_Z(d,t)
            C[(str(s),str(t))]=np.around(Tr(X.H*G),10)

    return C