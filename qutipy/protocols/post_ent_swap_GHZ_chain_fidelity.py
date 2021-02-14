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
from qutipy.fidelities import fidelity
from qutipy.general_functions import tensor


def post_ent_swap_GHZ_chain_fidelity(rho,n):

    '''
    Finds the fidelity of the output state of the apply_ent_swap_GHZ_chain_channel()
    function with respect to the (n+2)-party GHZ state.
    '''

    indices=list(itertools.product(*[range(2)]*n))

    f=0

    for index in indices:
        index=list(index)

        s=np.mod(sum(index),2)

        Bell_z=Bell_state(2,s,0,density_matrix=True)

        for z in index:
            Bell_z=tensor(Bell_z,Bell_state(2,z,0,density_matrix=True))

        f=f+fidelity(Bell_z,rho)

    return f