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

from qutipy.general_functions import trace_distance_pure_states,ket


def generate_state_2design(C,n,display=False):

    '''
    Takes the n-qubit Clifford gates provided in C and returns a
    corresponding state 2-design. This uses the fact that the Clifford
    gates (for any n) form a unitary 2-design, and that any unitary 
    t-design can be used to construct a state t-design.
    '''

    def in_list(L,elem):
    
        '''
        Checks if the given pure state elem is in the list L.
        '''
        
        x=0
        
        for l in L:
            if np.around(trace_distance_pure_states(l,elem),10)==0:
                x=1
                break
        
        return x

    S=[ket(2**n,0)]

    for c in C:
        s_test=c@ket(2**n,0)

        if not in_list(S,s_test):
            S.append(s_test)

        if display:
            print(len(S))

    return S