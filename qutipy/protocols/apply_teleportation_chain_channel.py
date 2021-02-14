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

from qutipy.general_functions import tensor,eye
from qutipy.states import Bell_state
from qutipy.Weyl import discrete_Weyl_X, discrete_Weyl_Z


def apply_teleportation_chain_channel(rho,n,dA=2,dR=2,dB=2):

    '''
    Applies the teleportation chain channel to the state rho, which is of the form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}.
    
    The channel is defined by performing a d-dimensional Bell basis measurement
    independently on the system pairs Ri1 and Ri2, for 1 <= i <= n; based on the
    outcome, a 'correction operation' is applied to B. The system pairs Ri1 and Ri2
    can be thought of as 'repeaters'. Note that n>=1. For n=1, we get the same channel
    as in apply_teleportation_channel().

    We obtain teleportation by letting dA=1 and letting

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B} = psi_{R11} ⊗ Phi_{R12 R21}^+ ⊗ ... ⊗ Phi_{Rn2 B}^+,
    
    so that we have teleportation of the state psi in the system R11 to the system B. 

    We obtain a chain of entanglement swaps by letting

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B} = Phi_{A R11}^+ ⊗ Phi_{R12 R21}^+ ⊗ ... ⊗ Phi_{Rn2 B}^+.
    '''

    indices=list(itertools.product(*[range(dB)]*n))

    rho_out=np.matrix(np.zeros((dA*dB,dA*dB),dtype=complex))

    for z_indices in indices:
        for x_indices in indices:

            Bell_zx=Bell_state(dB,z_indices[0],x_indices[0])
            for j in range(1,n):
                Bell_zx=tensor(Bell_zx,Bell_state(dB,z_indices[j],x_indices[j]))
            
            z_sum=np.mod(sum(z_indices),dB)
            x_sum=np.mod(sum(x_indices),dB)

            W_zx=(discrete_Weyl_Z(dB)**z_sum)*(discrete_Weyl_X(dB)**x_sum)

            rho_out=rho_out+tensor(eye(dA),Bell_zx.H,W_zx)*rho*tensor(eye(dA),Bell_zx,W_zx.H)

    return rho_out