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


from qutipy.su import su_structure_constants


def coherence_vector_star_product(n1,n2,d):

    '''
    Computes the star product between two coherence vectors corresponding to states, so that
    n1 and n2 (the coherence vectors) have length d^2-1 each.

    Definition taken from:
    
        "Characterization of the positivity of the density matrix in terms of
        the coherence vector representation"
        PHYSICAL REVIEW A 68, 062322 (2003)
    '''

    #L=su_generators(d)
    g=su_structure_constants(d)[1]

    p=[]

    for k in range(1,d**2):
        pk=0
        for i in range(1,d**2):
            for j in range(1,d**2):
                pk+=(d/2)*n1[i-1]*n2[j-1]*g[(i,j,k)]
        p.append(pk)

    return np.array(p)