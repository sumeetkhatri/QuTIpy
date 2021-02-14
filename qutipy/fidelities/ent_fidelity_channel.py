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

from qutipy.general_functions import Tr
from qutipy.states import MaxEnt_state
from qutipy.channels import Choi_representation



def ent_fidelity_channel(K,d):

    '''
    Finds the entanglement fidelity of the channel given by the set K of 
    Kraus operators. d is the dimension of the input space.
    '''

    Bell=MaxEnt_state(d)

    K_choi=(1./d)*Choi_representation(K,d)

    return np.real(Tr((Bell)*K_choi))