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

from qutipy.states import singlet_state
from qutipy.general_functions import SWAP,eye


def Werner_state(p,d,alt_param=False):

    '''
    Generates the Werner state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_W=p*singlet+(1-p)*singlet_perp,
        
    where singlet is the state defined as (1/(d^2-d))*(eye(d^2)-SWAP) and
    singlet_perp is the state defined as (1/(d^2+d))*(eye(d^2)+SWAP),
    where SWAP is the swap operator between two d-dimensional systems. The parameter
    p is between 0 and 1.
    
    Werner states are invariant under U ⊗ U for every unitary U.

    If alt_param=True, then the function returns a different parameterization of 
    the Werner state in which the parameter p is between -1 and 1, and

        rho_W=(1/(d^2-d*p))*(eye(d^2)-p*SWAP)

    '''
    
    if alt_param:
        F=SWAP([1,2],[d,d])
        return (1/(d**2-d*p))*(eye(d**2)-p*F)
    else:
        singlet,singlet_perp=singlet_state(d,perp=True)
        return p*singlet+(1-p)*singlet_perp