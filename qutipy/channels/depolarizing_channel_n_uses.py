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

from qutipy.general_functions import syspermute,eye,partial_trace,tensor


def depolarizing_channel_n_uses(p,n,rho,m):


    '''
    Generates the output state corresponding to the depolarizing channel
    applied to each one of n systems in the joint state rho. p is the 
    depolarizing probability as defined in the function "depolarizing_channel"
    above.

    If rho contains m>n systems, then the first m-n systems are left alone.
    '''

    dims=2*np.ones(m).astype(int)

    rho_out=np.zeros((2**m,2**m))

    for k in range(n+1):
        indices=list(itertools.combinations(range(1,n+1),k))

        #print k,indices

        for index in indices:
            index=list(index)

            index=np.array(index)+(m-n)
            index=list(index.astype(int))

            index_diff=np.setdiff1d(range(1,m+1),index)

            perm_arrange=np.append(index,index_diff).astype(int)
            perm_rearrange=np.zeros(m)

            for i in range(m):
                perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1

            perm_rearrange=perm_rearrange.astype(int)

            mix=eye(2**k)/2.**k

            rho_part=partial_trace(rho,index,dims)

            rho_out=rho_out+(4*p/3.)**k*(1-(4*p/3.))**(n-k)*syspermute(tensor(mix,rho_part),perm_rearrange,dims)

    return np.matrix(rho_out)