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


from qutipy.states import MaxEnt_state
from qutipy.channels import apply_channel



def Choi_representation(K,dimA):

    '''
    Calculates the Choi representation of the map with Kraus operators K.
    dimA is the dimension of the input space of the channel.

    The Choi represenatation is defined with the channel acting on the second
    half of the maximally entangled vector.
    '''


    #Gamma=np.sqrt(dimA)*MaxEnt_state(dimA)
    Gamma=MaxEnt_state(dimA,normalized=False)

    return np.matrix(apply_channel(K,Gamma,2,[dimA,dimA]),dtype=np.complex)