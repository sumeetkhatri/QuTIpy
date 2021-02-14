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

from qutipy.general_functions import tensor,eye,ket,partial_trace
from qutipy.gates import CNOT_ij, Rx_i



def apply_ent_swap_GHZ_chain_channel(rho,n):

    '''
    Applies the channel that takes n+1 copies of a maximally entangled state and outputs
    a (n+2)-party GHZ state. The input state rho is of the form

        rho_{A R11 R12 R21 R22 ... Rn1 Rn2 B}

    A CNOT is applies to each pair Rj1 Rj2. Then, the qubits Rj2 are measured in the
    standard basis. Conditioned on these outcomes, a correction operation is applied
    at B.

    Currently only works for qubits. For n=1, we get the same thing as apply_ent_swap_GHZ_channel().
    '''


    def K(j,x):

        # j is between 1 and n, denoting the pair of R systems. x is either 0 or 1.
        # For each j, the qubit indices are 2*j and 2*j+1 for the pair Rj1 and Rj2

        Mx=tensor(eye(2),eye(2**(2*j-2)),eye(2),ket(2,x)*ket(2,x).H,eye(2**(2*(n-j))),eye(2))
        
        C=CNOT_ij(2*j,2*j+1,2*n+2)

        X=1j*Rx_i(2*j+2,np.pi,2*n+2)

        return Mx*C*(X**x)


    indices=list(itertools.product(*[range(2)]*n))

    rho_out=np.matrix(np.zeros((2**(2*n+2),2**(2*n+2)),dtype=complex))

    for index in indices:
        index=list(index)

        L=K(1,index[0])
        for j in range(2,n+1):
            L=K(j,index[j-1])*L

        rho_out=rho_out+L*rho*L.H

    rho_out=partial_trace(rho_out,[2*j+1 for j in range(1,n+1)],[2]*(2*n+2))

    return rho_out