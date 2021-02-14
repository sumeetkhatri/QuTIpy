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

from qutipy.channels import amplitude_damping_channel


def generalized_amplitude_damping_channel(gamma,N):

    '''
    Generates the generalized amplitude damping channel.
    '''

    if N==0:
        return amplitude_damping_channel(gamma)
    elif N==1:
        A1=np.matrix([[np.sqrt(1-gamma),0],[0,1]])
        A2=np.matrix([[0,0],[np.sqrt(gamma),0]])
        return [A1,A2]
    else:
        A1=np.sqrt(1-N)*np.matrix([[1,0],[0,np.sqrt(1-gamma)]])
        A2=np.sqrt(1-N)*np.matrix([[0,np.sqrt(gamma)],[0,0]])
        A3=np.sqrt(N)*np.matrix([[np.sqrt(1-gamma),0],[0,1]])
        A4=np.sqrt(N)*np.matrix([[0,0],[np.sqrt(gamma),0]])

        return [A1,A2,A3,A4]