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

from qutipy.general_functions import SWAP,eye


def singlet_state(d):

    '''
    Generates the singlet state acting on two d-dimensional systems, which is defined
    as

        |Psi^-><Psi^-|=(1/(d^2-d))(eye(d^2)-F),

    where F is the swap operator given by SWAP([1,2],[d,d]) (see below).
    '''

    F=SWAP([1,2],[d,d])
    singlet=(1/(d**2-d))*(eye(d**2)-F)

    return singlet