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
from numpy.linalg import norm
from scipy.optimize import minimize

from qutipy.channels import apply_channel
from qutipy.entropies import coherent_inf_state



def coherent_inf_channel(K,dim_in,dim_out,s=1,display=True):

    '''
    Calculates the coherent information of the channel specified by
    the Kraus operators in K.

    If s=2, then calculates the reverse coherent information of the channel.
    '''


    def objfunc(x):

        Re=np.matrix(x[0:dim_in**2])
        Im=np.matrix(x[dim_in**2:])

        psi=np.matrix(Re.T+1j*Im.T)
        psi=psi/norm(psi)

        psi_AA=psi*psi.H

        rho_AB=apply_channel(K,psi_AA,2,dim=[dim_in,dim_in])

        return -coherent_inf_state(rho_AB,dim_in,dim_out,s)


    x_init=np.random.rand(2*dim_in**2)

    opt=minimize(objfunc,x_init,options={'disp':display})

    return np.max([0,-opt.fun])