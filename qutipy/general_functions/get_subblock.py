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


def get_subblock(X,sys,indices,dim):

    '''
    For the multipartite operator X, with local dimensions given by the list dim,
    this function extracts the subblock such that the systems in sys have the
    values given by those in indices. indices should be a list of tuples, each
    tuple corresponding to a system in sys. For each tuple, the first element
    gives the row and the second the column.

    For example, the subblock (<i| ⊗ id ⊗ <j|)X(|k> ⊗ id ⊗ |l>)
    is given by get_subblock(X,[1,3],[(i,k),(j,l)],[dim1,dim2,dim3]), so the result
    is a dim2 x dim2 matrix, which is the desired subblock.
    '''

    X=np.array(X)

    X_reshape=np.reshape(X,dim+dim)

    num_sys=len(dim)
    total_sys=range(1,num_sys+1)
    dims_keep=[dim[s-1] for s in list(set(total_sys)-set(sys))]

    to_slice=[slice(None) for i in range(2*num_sys)]

    count=0
    for s in total_sys:
        if s in sys:
            to_slice[s-1]=indices[count][0]
            to_slice[num_sys+s-1]=indices[count][1]
            count+=1
    
    X_reshape=X_reshape[to_slice]

    X_reshape=np.reshape(X_reshape,dims_keep+dims_keep)

    return X_reshape