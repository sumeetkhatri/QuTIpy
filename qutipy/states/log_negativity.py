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

from qutipy.general_functions import partial_transpose,trace_norm


def log_negativity(rhoAB,dimA,dimB):

    '''
    Returns the log-negativity of the bipartite state rhoAB, which is defined as

        E_N(rhoAB) = log_2 || rhoAB^{T_B} ||_1.

    This is a faithful entanglement measure if both A and B are qubits or if one of
    then is a qubit and the other a qutrit. Such states are entangled if and only if
    the log-negativity is positive.

    See "Computable measure of entanglement", Phys. Rev. A 65, 032314 (2002).
    '''

    return np.log2(trace_norm(partial_transpose(rhoAB,[2],[dimA,dimB])))