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
from qutipy.states import Bell_state
from qutipy.general_functions import tensor


def post_ent_swap_GHZ_fidelity(rho):

    # Last modified: 15 June 2020

    '''
    Finds the fidelity of the output state of the apply_ent_swap_GHZ_channel() function
    with respect to the three-party GHZ state.
    '''

    Phi=[Bell_state(2,z,0,density_matrix=True) for z in range(2)]

    return sum([fidelity(tensor(Phi[z].H,Phi[z].H),rho) for z in range(2)])