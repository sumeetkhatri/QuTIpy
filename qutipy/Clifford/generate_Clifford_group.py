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

from qutipy.Clifford import Clifford_group_generators
from qutipy.general_functions import unitary_distance,eye


def generate_Clifford_group(n,display=False):
    
    '''
    Generates the n-qubit Clifford group. The display variable is for testing
    purposes, and to see the progress through the code.

    Note that even for n=2, this code will take a long time to run! There are 
    11520 elements of the two-qubit Clifford group!
    '''

    G=Clifford_group_generators(n)
    
    def in_list(L,elem):
    
        # Last modified: 27 June 2019
        
        '''
        Checks if the given unitary elem is in the list L.
        '''
        
        x=0
        for l in L:
            if np.around(unitary_distance(l,elem),10)==0:  # Check of the distance is zero (up to 10 decimal places)
                x=1
                break
        
        return x


    C=[eye(2**n)]
    generated=False

    while not generated:

        tmp=[]
        num_added=0

        for c in C:
            for g in G:
                t1=c*g
                t2=c*g.H
                
                # t1 and t2 might be the same, in which case we add only one of the two to the list (if needed).
                # Also, t1 and t2 might already by in tmp (up to global phase), so we need to check for that as well.
                if np.around(unitary_distance(t1,t2),10)==0:
                    if not in_list(C,t1) and not in_list(tmp,t1):
                        tmp.append(t1)
                        num_added+=1
                else:  # if t1 and t2 are different, add both to the list (if needed).
                    if not in_list(C,t1) and not in_list(tmp,t1):
                        tmp.append(t1)
                        num_added+=1
                    if not in_list(C,t2) and not in_list(tmp,t2):
                        tmp.append(t2)
                        num_added+=1
        
        if num_added>0:
            for t in tmp:
                C.append(t)
        else:
            generated=True

        if display:
            print(len(C))
            
    return C