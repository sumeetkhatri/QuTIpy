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


def cvxpy_to_numpy(cvx_obj):

    '''
    Converts a cvxpy variable into a numpy array.
    '''

    if cvx_obj.is_scalar():
        return np.array(cvx_obj)
    elif len(cvx_obj.shape)==1:  # cvx_obj is a (column or row) vector
        return np.array(list(cvx_obj))
    else:  # cvx_obj is a matrix
        X=[]
        for i in range(cvx_obj.shape[0]):
            x=[cvx_obj[i,j] for j in range(cvx_obj.shape[1])]
            X.append(x)
        X=np.array(X)
        return X
