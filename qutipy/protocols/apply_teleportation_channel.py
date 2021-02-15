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
from numpy.linalg import matrix_power

from qutipy.Weyl import discrete_Weyl_X, discrete_Weyl_Z
from qutipy.general_functions import dag,eye,tensor
from qutipy.states import Bell_state



def apply_teleportation_channel(rho,dA=2,dR1=2,dR2=2,dB=2):

    '''
    Applies the d-dimensional teleportation channel to the four-qudit state rho_{AR1R2B}.
    The channel measures R1 and R2 in the d-dimensional Bell basis and, based on the
    outcome, applies a 'correction operation' to B. So the output of the channel consists
    only of the systems A and B.

    We obtain quantum teleportation by letting

        rho_{AR1R2B} = psi_{R1} ⊗ Phi_{R2B}^+,

    so that dA=1. This simulates teleportation of the state psi in the system R1 to
    the system B. 

    We obtain entanglement swapping by letting

        rho_{AR1R2B} = Phi_{AR1}^+ ⊗ Phi_{R2B}^+.
    
    The result of the channel is then Phi_{AB}^+
    '''

    X=[matrix_power(discrete_Weyl_X(dB),x) for x in range(dB)]
    Z=[matrix_power(discrete_Weyl_Z(dB),z) for z in range(dB)]
    
    rho_out=np.sum([tensor(eye(dA),dag(Bell_state(dR1,z,x)),Z[z]@X[x])@rho@tensor(eye(dA),Bell_state(dR1,z,x),dag(X[x])@dag(Z[z])) for z in range(dB) for x in range(dB)],0)

    return rho_out