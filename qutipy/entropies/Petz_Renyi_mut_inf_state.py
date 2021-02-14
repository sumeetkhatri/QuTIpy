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



from qutipy.general_functions import partial_trace,tensor
from qutipy.entropies import Petz_Renyi_rel_ent


def Petz_Renyi_mut_inf_state(rhoAB,dimA,dimB,alpha,opt=True):

    '''
    Computes the Petz-Renyi mutual information of the bipartite state
    rhoAB for 0<=alpha<=1.

    TO DO: Figure out how to do the computation with optimization over sigmaB.
    '''

    rhoA=partial_trace(rhoAB,[2],[dimA,dimB])
    rhoB=partial_trace(rhoAB,[1],[dimA,dimB])

    if opt==False:
        return Petz_Renyi_rel_ent(rhoAB,tensor(rhoA,rhoB),alpha)
    else:
        return None