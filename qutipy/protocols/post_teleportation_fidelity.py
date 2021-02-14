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


from qutipy.fidelities import fidelity
from qutipy.general_functions import tensor
from qutipy.states import Bell_state


def post_teleportation_fidelity(rho,dA=2):

    '''
    Calculates the fidelity of the output state of the teleportation channel with
    respect to the maximally entangled state on AB. The input state rho is of the
    form rho_{AR1R2B}. We assume that A, R1, R2, B all have the same dimension.
    '''

    return sum([fidelity(rho,tensor(Bell_state(dA,z,x,density_matrix=True),Bell_state(dA,z,x,density_matrix=True))) for z in range(dA) for x in range(dA)])
