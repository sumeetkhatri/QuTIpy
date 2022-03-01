'''
This code is part of QuTIpy.

(c) Copyright Sumeet Khatri, 2022

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

    For example, ket(2,0)=|0> and ket(2,1)=|1>.

    In general, ket(d,j), for j between 0 and d-1, generates a column vector
    (as a numpy matrix) in which the jth element is equal to 1 and the rest
    are equal to zero.

    ket(d,[j1,j2,...,jn]) generates the tensor product |j1>|j2>...|jn> of
    d-dimensional basis vectors.

    If dim is specified as a list, then, e.g., ket([d1,d2],[j1,j2]) generates the
    tensor product |j1>|j2>, with the first tensor factor being d1-dimensional
    and the second tensor factor being d2-dimensional.
    '''

    args=np.array(args)

    if args.size==1:
        num=args[0]
        out=np.zeros([dim,1])
        out[num]=1
    else:
        args=args[0]
        if type(dim)==int:
            out=ket(dim,args[0])
            for j in range(1,len(args)):
                out=np.kron(out,ket(dim,args[j]))
        elif type(dim)==list:
            out=ket(dim[0],args[0])
            for j in range(1,len(args)):
                out=np.kron(out,ket(dim[j],args[j]))
    
    return out