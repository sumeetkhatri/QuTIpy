'''
This code is part of QuTIpy.

(c) Copyright Sumeet Khatri, 2022

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
'''


import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm

from qutipy.general_functions import dag
from qutipy.entropies import entropy
from qutipy.channels import apply_channel


def min_output_entropy(K,dim,display=True):

    '''
    Computes the minimum output entropy of a channel given by its set of
    Kraus operators K. dim is the dimension of the input space of the
    channel.
    '''


    def objfunc(x):

        Re=np.array(x[0:dim])
        Im=np.array(x[dim:])

        psi=np.array([Re+1j*Im]).T
        psi=psi/norm(psi)

        rho=psi@dag(psi)
        rho_out=apply_channel(K,rho)

        return entropy(rho_out)

    
    x_init=np.random.rand(2*dim)

    opt=minimize(objfunc,x_init,options={'disp':display})

    return opt.fun