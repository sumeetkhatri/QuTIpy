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

from qutipy.states import Bell_state
from qutipy.general_functions import tensor
from qutipy.fidelities import fidelity



def post_teleportation_chain_fidelity(rho,n,dA=2):

    '''
    Calculates the fidelity of the output state of the teleportation chain channel with
    respect to the maximally entangled state on AB. The input state rho is of the
    form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}.

    We assume that A, B, and all R systems have the same dimension.
    '''

    f=0

    indices=list(itertools.product(*[range(dA)]*n))

    for z_indices in indices:
        for x_indices in indices:

            z_sum=np.mod(sum(z_indices),dA)
            x_sum=np.mod(sum(x_indices),dA)

            Bell_tot=Bell_state(dA,z_sum,x_sum,density_matrix=True)

            for j in range(n):
                Bell_tot=tensor(Bell_tot,Bell_state(dA,z_indices[j],x_indices[j],density_matrix=True))

            f+=fidelity(rho,Bell_tot)

    return f