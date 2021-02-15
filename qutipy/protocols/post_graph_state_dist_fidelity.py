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
import itertools

from qutipy.general_functions import tensor,syspermute
from qutipy.states import Bell_state
from qutipy.fidelities import fidelity



def post_graph_state_dist_fidelity(A_G,n,rho):

    '''
    Finds the fidelity of the output state of the graph state distribution channel
    with respect to the graph state |G>, where A_G is the adjacency matrix of the 
    graph G and n is the number of vertices of G.
    '''

    X_n=list(itertools.product(*[range(2)]*n))

    f=0

    for x_n in X_n:

        x_n=np.array([x_n]).T  # Turn x_n into a column vector matrix

        z_n=A_G*x_n
        z_n=np.mod(z_n,2)

        Bell=Bell_state(2,z_n[0,0],x_n[0,0],density_matrix=True)
    
        for k in range(1,n):
            Bell=tensor(Bell,Bell_state(2,z_n[k,0],x_n[k,0],density_matrix=True))
        
        Bell=syspermute(Bell,list(range(1,2*n,2))+list(range(2,2*n+1,2)),2*np.ones(2*n,dtype=int))

        f+=fidelity(rho,Bell)

    return f