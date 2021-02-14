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
from numpy.linalg import norm

from qutipy.linalg import proj


def gram_schmidt(states,dim,normalize=True):

    '''
    Performs the Gram-Schmidt orthogonalization procedure on the given states
    (or simply vectors). dim is the dimension of the vectors.
    '''

    e=[]
    u=[]
    u.append(states[0])
    e.append(states[0]/norm(states[0]))

    for k in range(1,len(states)):
        S=np.matrix(np.zeros([dim,1]),dtype=complex)
        for j in range(k):
            S+=proj(u[j],states[k])
        u.append(states[k]-S)
        e.append(u[k]/norm(u[k]))
    
    if normalize==True:
        return e
    else:
        return u