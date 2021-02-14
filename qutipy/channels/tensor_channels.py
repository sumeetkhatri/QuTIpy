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


import itertools

from qutipy.general_functions import tensor


def tensor_channels(C):

    '''
    Takes the tensor product of the channels in C.

    C is a set of sets of Kraus operators.
    '''

    lengths=[]
    for c in C:
        lengths.append(len(c))
    
    combs=list(itertools.product(*[range(length) for length in lengths]))

    K_n=[]

    for comb in combs:
        tmp=1
        for i in range(len(comb)):
            tmp=tensor(tmp,C[i][comb[i]])
        K_n.append(tmp)

    return K_n