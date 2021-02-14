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

from qutipy.states import MaxEnt_state
from qutipy.general_functions import eye


def isotropic_state(p,d,fidelity=False):

    '''
    Generates the isotropic state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_Iso = p*|Bell><Bell|+(1-p)*eye(d^2)/d^2,

    where -1/(d^2-1)<=p<=1. Isotropic states are invariant under U ⊗ conj(U)
    for any unitary U, where conj(U) is the complex conjugate of U.

    If fidelity=True, then the function returns a different parameterization of
    the isotropic state in which the parameter p is the fidelity of the state
    with respect to the maximally entangled state.
    '''

    Bell=MaxEnt_state(d)

    if fidelity:
        return p*Bell+((1-p)/(d**2-1))*(eye(d**2)-Bell)
    else:
        return p*Bell+(1-p)*eye(d**2)/d**2