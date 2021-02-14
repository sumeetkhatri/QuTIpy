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

from qutipy.general_functions import tensor,ket,eye
from qutipy.gates import CZ_ij



def graph_state(A_G,n,density_matrix=False,return_CZ=False):

    '''
    Generates the graph state corresponding to the undirected graph G with n vertices.
    A_G denotes the adjacency matrix of G, which for an undirected graph is a binary
    symmetric matrix indicating which vertices are connected.

    See the following book chapter for a review:

        ``Cluster States'' in Compedium of Quantum Physics, pp. 96-105, by H. J. Briegel.

    '''

    plus=(1/np.sqrt(2))*(ket(2,0)+ket(2,1))

    plus_n=tensor([plus,n])

    CZ_G=eye(2**n)

    for i in range(n):
        for j in range(i,n):
            if A_G[i,j]==1:
                CZ_G=CZ_G*CZ_ij(i+1,j+1,n)

    if density_matrix:
        plus_n=plus_n*plus_n.H
        if return_CZ:
            return CZ_G*plus_n*CZ_G.H,CZ_G
        else:
            return CZ_G*plus_n*CZ_G.H
    else:
        if return_CZ:
            return CZ_G*plus_n,CZ_G
        else:
            return CZ_G*plus_n