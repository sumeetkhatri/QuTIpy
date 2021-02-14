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

from qutipy.Weyl import discrete_Weyl_Z
from qutipy.channels import Pauli_channel


def dephasing_channel(p,d=2):

    '''
    Generates the channel rho -> (1-p)*rho+p*Z*rho*Z. (In the case d=2.)

    For d>=2, we let p be a list of d probabilities, and we use the discrete Weyl-Z
    operators to define the channel.

    For p=1/d, we get the completely dephasing channel.
    '''

    if d==2:
        return Pauli_channel(0,0,p)
    else:
        K=[np.sqrt(p[k])*discrete_Weyl_Z(d)**k for k in range(d)]
        return K