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


from qutipy.entropies import entropy


def Holevo_inf_ensemble(p,S):

    '''
    Computes the Holevo information of an ensemble.

    p is an array of probabilities, and S is an array of states.

    Based on MATLAB code written by Felix Lediztky.
    '''

    dim=np.shape(S[0])[0]

    R=np.matrix(np.zeros((dim,dim)))
    av=0

    for i in range(len(p)):
        R=R+p[i]*S[i]
        av=av+p[i]*entropy(S[i])

    return entropy(R)-av