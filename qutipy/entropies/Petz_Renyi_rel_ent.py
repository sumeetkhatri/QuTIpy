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
from scipy.linalg import fractional_matrix_power

from qutipy.general_functions import Tr



def Petz_Renyi_rel_ent(rho,sigma,alpha):

    '''
    Computes the Petz-Renyi relative entropy of rho and sigma for 0<=alpha<=1.
    '''

    rho_a=fractional_matrix_power(rho,alpha)
    sigma_a=fractional_matrix_power(sigma,1-alpha)

    Q=np.real(Tr(rho_a@sigma_a))

    return (1./(alpha-1))*np.log2(Q)