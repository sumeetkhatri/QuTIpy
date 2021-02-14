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


from qutipy.gates import H_i,S_i,CNOT_ij


def Clifford_group_generators(n):

    '''
    Outputs the generators of the n-qubit Clifford group.
    '''

    G=[]

    if n==1:
        G=[H_i(1,1),S_i(1,1)]
    else:
        for i in range(1,n+1):
            G.append(H_i(i,n))
            G.append(S_i(i,n))
            for j in range(1,n+1):
                if i<j:
                    G.append(CNOT_ij(i,j,n))
                else:
                    continue
    
    return G