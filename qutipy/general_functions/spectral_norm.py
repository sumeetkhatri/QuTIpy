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


from numpy.linalg import norm


def spectral_norm(X):

    '''
    Finds the spectral norm (also known as the operator norm and the Schatten
    infinity-norm) of the matrix X. (The largest singular value of X.)
    '''

    return norm(X,ord=2)