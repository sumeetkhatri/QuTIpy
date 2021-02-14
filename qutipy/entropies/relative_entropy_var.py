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

from qutipy.entropies import relative_entropy



def relative_entropy_var(rho,sigma):

    '''
    Returns the relative entropy variance of rho and sigma, defined as

    V(rho||sigma)=Tr[rho*(log2(rho)-log2(sigma))^2]-D(rho||sigma)^2.
    '''

    return np.real(np.trace(rho*(np.matrix(logm(rho))/np.log(2)-np.matrix(logm(sigma))/np.log(2))**2))-relative_entropy(rho,sigma)**2