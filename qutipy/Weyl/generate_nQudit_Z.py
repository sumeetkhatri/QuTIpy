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


from numpy.linalg import matrix_power

from qutipy.Weyl import discrete_Weyl_Z
from qutipy.general_functions import tensor


def generate_nQudit_Z(d,indices):

    '''
    Generates a tensor product of discrete Weyl-Z operators. indices is a
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    '''

    Z=discrete_Weyl_Z(d)

    out=1

    for index in indices:
        out=tensor(out,matrix_power(Z,index))

    return out