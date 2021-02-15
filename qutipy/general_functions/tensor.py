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

import numpy as np


def tensor(*args):

    '''
    Takes the tensor product of an arbitrary number of matrices/vectors.
    '''

    M=1

    for j in range(len(args)):
        if isinstance(args[j],list):
            for k in range(args[j][1]):
                M=np.kron(M,args[j][0])
        else:
            M=np.kron(M,args[j])

    return M