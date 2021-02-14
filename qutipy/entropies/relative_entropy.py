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
from scipy.linalg import logm


def relative_entropy(rho,sigma):

    '''
    Computes the standard (von Neumann) quantum relative entropy of rho
    and sigma, provided that supp(rho) is contained in supp(sigma).
    '''

    return np.real(np.trace(rho*(np.matrix(logm(rho))-np.matrix(logm(sigma)))))/np.log(2)
