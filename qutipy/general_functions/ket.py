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


def ket(dim,*args):

    '''
    Generates a standard basis vector in dimension dim.

    For example, ket(2,0)=|0>=[1,0] and ket(2,1)=|1>=[0,1].

    In general, ket(d,j), for j between 0 and d-1, generates a column vector
    (as a numpy matrix) in which the jth element is equal to 1 and the rest
    are equal to zero.

    ket(d,[j_1,j_2,...,j_n]) generates the tensor product |j_1>|j_2>...|j_n> of
    d-dimensional basis vectors.
    '''

    args=np.array(args)

    if args.size==1:
        num=args[0]
        out=np.zeros([dim,1])
        out[num]=1
    else:
        args=args[0]
        out=ket(dim,args[0])
        for j in range(1,len(args)):
            out=np.kron(out,ket(dim,args[j]))
    
    return np.matrix(out)