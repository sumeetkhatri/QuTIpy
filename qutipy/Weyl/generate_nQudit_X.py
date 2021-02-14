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


from qutipy.Weyl import discrete_Weyl_X
from qutipy.general_functions import tensor


def generate_nQudit_X(d,indices):

    '''
    Generates a tensor product of discrete Weyl-X operators. indices is a 
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    '''

    X=discrete_Weyl_X(d)

    out=1

    for index in indices:
        out=tensor(out,X**index)

    return out