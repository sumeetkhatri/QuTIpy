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


from qutipy.general_functions import tensor,partial_trace
from qutipy.entropies import relative_entropy



def mutual_information(rhoAB,dimA,dimB):

    '''
    Computes the mutual information of the bipartite state rhoAB, defined as

    I(A;B)_rho=D(rhoAB||rhoA\otimes rhoB)
    '''

    rhoA=partial_trace(rhoAB,[2],[dimA,dimB])
    rhoB=partial_trace(rhoAB,[1],[dimA,dimB])

    return relative_entropy(rhoAB,tensor(rhoA,rhoB))