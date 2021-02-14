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
from scipy.optimize import minimize
from numpy.linalg import norm

from qutipy.general_functions import tensor,eye,ket
from qutipy.entropies import Holevo_inf_ensemble
from qutipy.channels import apply_channel


def Holevo_inf_channel(K,dim,display=True):

    '''
    Computes the Holevo information of a channel given by its set of
    Kraus operators K. dim is the dimension of the input space of the
    channel.

    Based on MATLAB code written by Felix Leditzky.
    '''


    def objfunc(x):

        Re=np.matrix(x[0:dim**3])
        Im=np.matrix(x[dim**3:])

        psi=np.matrix(Re.T+1j*Im.T)
        psi=psi/norm(psi)

        p=[]
        S=[]

        for j in range(dim**2):
            R=tensor(ket(dim**2,j),eye(dim)).H*(psi*psi.H)*tensor(ket(dim**2,j),eye(dim))
            p.append(np.trace(R))
            rho=R/np.trace(R)
            rho_out=apply_channel(K,rho)
            S.append(rho_out)
        
        return -np.real(Holevo_inf_ensemble(p,S))

    
    x_init=np.random.rand(2*dim**3)

    opt=minimize(objfunc,x_init,options={'disp':display})

    return -opt.fun