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


def n_channel_uses(K,n):

    '''
    Given the Kraus operators K of a channel, this function generates the
    Kraus operators corresponding to the n-fold tensor power of the channel.
    dimA is the dimension of the input space, and dimB the dimension of the
    output space.
    '''

    r=len(K)   # Number of Kraus operators

    combs=list(itertools.product(*[range(r)]*n))

    K_n=[]

    for comb in combs:
        #print comb
        tmp=1
        for i in range(n):
            tmp=tensor(tmp,K[comb[i]])
        K_n.append(tmp)

    return K_n