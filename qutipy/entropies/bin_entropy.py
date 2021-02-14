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


def bin_entropy(p):

    '''
    Returns the binary entropy for 0<=p<=1.
    '''

    if p==0:
        return 0
    elif p==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)