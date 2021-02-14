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
from scipy.linalg import fractional_matrix_power


def sandwiched_Renyi_rel_ent(rho,sigma,alpha):

    '''
    Computes the sandwiched Renyi relative entropy for either 0<=alpha<=1,
    or for alpha>=1 provided that supp(rho) is contained in supp(sigma).
    '''

    sigma_a=np.matrix(fractional_matrix_power(sigma,(1.-alpha)/(2*alpha)))

    Q=np.real(np.trace(fractional_matrix_power(sigma_a*rho*sigma_a,alpha)))

    return (1./(alpha-1))*np.log2(Q)