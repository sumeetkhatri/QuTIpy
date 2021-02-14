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

def phase_damping_channel(p):

    '''
    Generates the phase damping channel.
    '''

    K1=np.matrix([[1,0],[0,np.sqrt(p)]])
    K2=np.matrix([[0,0],[0,np.sqrt(1-p)]])

    return [K1,K2]

    