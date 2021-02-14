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


from qutipy.fidelities import ent_fidelity
from qutipy.channels import Choi_representation



def avg_fidelity(K,dimA):

    '''
    Calculates the average fidelity of a channel using its entanglement
    fidelity with respect to the maximally mixed state (see, e.g., Eq.
    (9.245) of Wilde's book.)

    K is the set of Kraus operators of the channel, and dimA is the dimension
    of the input space of the channel.
    '''

    choi_state=(1./dimA)*Choi_representation(K,dimA)

    return (dimA*ent_fidelity(choi_state,dimA)+1)/(dimA+1)