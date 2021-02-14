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


from qutipy.general_functions import partial_trace
from qutipy.entropies import entropy


def coherent_inf_state(rho_AB,dimA,dimB,s=1):

    '''
    Calculates the coherent information of the state rho_AB.

    If s=2, then calculates the reverse coherent information.
    '''

    if s==1: # Calculate I_c(A>B)=H(B)-H(AB)
        rho_B=partial_trace(rho_AB,[1],[dimA,dimB])
        return entropy(rho_B)-entropy(rho_AB)
    else: # Calculate I_c(B>A)=H(A)- H(AB) (AKA reverse coherent information)
        rho_A=partial_trace(rho_AB,[2],[dimA,dimB])
        return entropy(rho_A)-entropy(rho_AB)