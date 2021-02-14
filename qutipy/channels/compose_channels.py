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



import itertools


def compose_channels(C):

    '''
    Takes a composition of channels. The variable C should be a list of lists,
    with each list consisting of the Kraus operators of the channels to be composed.

    If C=[K1,K2,...,Kn], then this function returns the composition such that
    the channel corresponding to K1 is applied first, then K2, etc.
    '''

    lengths=[]
    for c in C:
        lengths.append(len(c))
    
    combs=list(itertools.product(*[range(length) for length in lengths]))

    K_n=[]

    for comb in combs:
        tmp=1
        for i in range(len(comb)):
            tmp=C[i][comb[i]]*tmp
        K_n.append(tmp)

    return K_n