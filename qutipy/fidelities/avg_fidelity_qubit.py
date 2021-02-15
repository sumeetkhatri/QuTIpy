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

from qutipy.general_functions import dag,ket,Tr
from qutipy.channels import apply_channel



def avg_fidelity_qubit(K):

    '''
    K is the set of Kraus operators for the (qubit to qubit) channel whose
    average fidelity is to be found.
    '''

    ket0=ket(2,0)
    ket1=ket(2,1)
    ket_plus=(1./np.sqrt(2))*(ket0+ket1)
    ket_minus=(1./np.sqrt(2))*(ket0-ket1)
    ket_plusi=(1./np.sqrt(2))*(ket0+1j*ket1)
    ket_minusi=(1./np.sqrt(2))*(ket0-1j*ket1)

    states=[ket0,ket1,ket_plus,ket_minus,ket_plusi,ket_minusi]

    F=0

    for state in states:

        F+=np.real(Tr((state@dag(state))*apply_channel(K,state@dag(state))))

    return (1./6.)*F