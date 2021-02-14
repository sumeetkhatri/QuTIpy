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


def BB84_channel(Q):

    '''
    Generates the channel corresponding to the BB84 protocol with
    equal X and Z errors, given by the QBER Q. The definition of this
    channel can be found in:

        "Additive extensions of a quantum channel", by
            Graeme Smith and John Smolin. (arXiv:0712.2471)

    '''

    return Pauli_channel(Q-Q**2,Q**2,Q-Q**2)