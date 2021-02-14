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

from qutipy.gates import CNOT_ij
from qutipy.general_functions import tensor,eye,ket
from qutipy.Weyl import discrete_Weyl_X


def apply_ent_swap_GHZ_channel(rho):

    # Last modified: 15 June 2020

    '''
    Applies the channel that takes two copies of a maximally entangled state and outputs
    a three-party GHZ state. The input state rho is of the form

        rho_{A R1 R2 B}.

    A CNOT is applied to R1 and R2, followed by a measurement in the standard basis on
    R2, followed by a correction operation on B based on the outcome of the measurement.

    Currently only works for qubits.
    '''

    C=CNOT_ij(2,3,4)

    X=[discrete_Weyl_X(2)**x for x in range(2)]
    
    rho_out=np.matrix(np.sum([tensor(eye(4),ket(2,x).H,eye(2))*C*tensor(eye(8),X[x])*rho*tensor(eye(8),X[x])*C.H*tensor(eye(4),ket(2,x),eye(2)) for x in range(2)],0))

    return rho_out