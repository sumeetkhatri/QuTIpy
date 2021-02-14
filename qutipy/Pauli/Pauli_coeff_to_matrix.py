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

from qutipy.Pauli import generate_nQubit_Pauli


def Pauli_coeff_to_matrix(coeffs,n):

    '''
    Takes the coefficients of a matrix in the n-qubit Pauli basis and outputs it
    as a matrix.

    coeffs should be specified as a one-dimensional list or array in standard
    lexicographical ordering.
    '''

    all_indices=list(itertools.product(*[range(0,4)]*n))

    out=0+0j

    for i in range(len(all_indices)):
        out+=(1./2.**n)*coeffs[i]*generate_nQubit_Pauli(all_indices[i])

    return out