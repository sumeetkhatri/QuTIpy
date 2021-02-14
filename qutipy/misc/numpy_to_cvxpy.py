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


from cvxpy import bmat


def numpy_to_cvxpy(np_obj):

    '''
    Converts numpy array to cvxpy expression.
    '''

    np_obj_list=np_obj.tolist()
    
    return bmat(np_obj_list)

