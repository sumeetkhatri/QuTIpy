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


from qutipy.states import MaxEnt_state
from qutipy.general_functions import eye,Tr


def isotropic_twirl_state(rho,d):

    '''
    Applies the twirling channel

        rho -> ∫ (U ⊗ conj(U))*rho*(U ⊗ conj(U)).H dU

    to the input state rho acting on two d-dimensional systems.

    For d=2, this is equivalent to

        rho -> (1/24)*sum_i (c_i ⊗ conj(c_i))*rho*(c_i ⊗ conj(c_i)).H

    where the unitaries c_i form the one-qubit Clifford group (because the Clifford
    unitaries constitute a unitary 2-design).

    This channel takes any state rho and converts it to an isotropic state with
    the same fidelity to the maximally entangled state as rho.
    '''

    #f=fidelity(rho,MaxEnt_state(d)*MaxEnt_state(d).H)
    #return isotropic_state(f,d,fidelity=True)

    G=MaxEnt_state(d,normalized=False,density_matrix=True)

    return (Tr(rho)/(d**2-1)-Tr(G*rho)/(d*(d**2-1)))*eye(d**2)+(Tr(G*rho)/(d**2-1)-Tr(rho)/(d*(d**2-1)))*G
