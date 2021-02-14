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
import itertools

from qutipy.general_functions import tensor
from qutipy.states import graph_state
from qutipy.Pauli import generate_nQubit_Pauli_Z



def apply_graph_state_dist_channel(A_G,n,rho):

    '''
    Applies the graph state distribution channel to the 2n-partite state rho, where
    n is the number of vertices in the graph G with adjacency matrix A_G (binary 
    symmetric matrix).

    rho is a state of the form rho_{A_1...A_n R_1...R_n}

    The local graph state operations and measurements are applied to the qubits
    R_1,...,R_n, and the correction operations are applied to A_1,...,A_n.

    When rho is a state of the form

        Phi_{A_1 R_1}^+ ⊗ Phi_{A_2 R_2}^+ ⊗ ... ⊗ Phi_{A_n R_n}^+,

    then the output state on A_1,...,A_n is the graph state |G>. 
    '''

    indices=list(itertools.product(*[range(2)]*n))

    H=(1/np.sqrt(2))*np.matrix([[1,1],[1,-1]])
    Hn=tensor([H,n])

    ket_G=graph_state(A_G,n)

    rho_out=np.matrix(np.zeros((2**n,2**n),dtype=complex))

    for index in indices:
        Zx=generate_nQubit_Pauli_Z(index)

        Gx=Zx*ket_G
        rho_out=rho_out+tensor(Zx,Gx.H)*rho*tensor(Zx,Gx)

    
    return rho_out