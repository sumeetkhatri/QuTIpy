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


from qutipy.channels import Pauli_channel


def bit_flip_channel(p):

    '''
    Generates the channel rho -> (1-p)*rho+p*X*rho*X. 
    '''

    return Pauli_channel(p,0,0)