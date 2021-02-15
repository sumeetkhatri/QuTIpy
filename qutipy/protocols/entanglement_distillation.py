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


from qutipy.gates import CNOT_ij
from qutipy.general_functions import dag,ket,eye,tensor,partial_trace,syspermute,Tr
from qutipy.states import isotropic_twirl_state



def entanglement_distillation(rho1,rho2,outcome=1,twirl_after=False,normalize=False):

    '''
    Applies a particular entanglement distillation channel to the two two-qubit states
    rho1 and rho2. [PRL 76, 722 (1996)]

    The channel is probabilistic. If the variable outcome=1, then the function returns
    the two-qubit state conditioned on the success of the distillation protocol.
    '''

    CNOT=CNOT_ij(1,2,2)
    proj0=ket(2,0)@dag(ket(2,0))
    proj1=ket(2,1)@dag(ket(2,1))

    P0=tensor(eye(2),proj0,eye(2),proj0)
    P1=tensor(eye(2),proj1,eye(2),proj1)
    P2=eye(16)-P0-P1
    C=tensor(CNOT,CNOT)
    K0=P0*C
    K1=P1*C
    K2=P2*C

    rho_in=syspermute(tensor(rho1,rho2),[1,3,2,4],[2,2,2,2]) # rho_in==rho_{A1A2B1B2}

    if outcome==1:
        # rho_out is unnormalized. The trace of rho_out is equal to the success probability.
        rho_out=partial_trace(K0@rho_in@dag(K0)+K1@rho_in@dag(K1),[2,4],[2,2,2,2])
        if twirl_after:
            rho_out=isotropic_twirl_state(rho_out,2)
        if normalize:
            rho_out=rho_out/Tr(rho_out)

    elif outcome==0:
        # rho_out is unnormalized. The trace of rho_out is equal to the failure probability.
        rho_out=partial_trace(K2@rho_in@dag(K2),[2,4],[2,2,2,2])
        if normalize:
            rho_out=rho_out/Tr(rho_out)

    return rho_out