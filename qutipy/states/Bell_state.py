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


from qutipy.states import MaxEnt_state
from qutipy.general_functions import tensor,eye
from qutipy.Weyl import discrete_Weyl_X, discrete_Weyl_Z


def Bell_state(d,z,x,density_matrix=False):

    '''
    Generates a d-dimensional Bell state with 0 <= z,x <= d-1. These are defined as

    |Phi_{z,x}> = (Z(z)X(x) ⊗ I)|Phi^+>

    '''

    Bell=MaxEnt_state(d,density_matrix=density_matrix)

    W_zx=discrete_Weyl_Z(d)**z*discrete_Weyl_X(d)**x

    if density_matrix:
        out=tensor(W_zx,eye(d))*Bell*tensor(W_zx.H,eye(d))
        return out
    else:
        out=tensor(W_zx,eye(d))*Bell
        return out