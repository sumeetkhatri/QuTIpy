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


def channel_scalar_multiply(K,x):

    '''
    Multiplies the channel with Kraus operators in K by the scalar x.
    This means that each Kraus operator is multiplied by sqrt(x)!
    '''

    K_new=[]
    
    for i in range(len(K)):
        K_new.append(np.sqrt(x)*K[i])

    return K_new