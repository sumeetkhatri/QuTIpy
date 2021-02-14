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
from scipy.linalg import sqrtm

from qutipy.general_functions import trace_norm


def fidelity(rho,sigma):

    '''
    Returns the fidelity between the states rho and sigma.
    '''

    return trace_norm(np.matrix(sqrtm(rho))*np.matrix(sqrtm(sigma)))**2