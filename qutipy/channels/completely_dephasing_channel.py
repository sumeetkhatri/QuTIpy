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

from qutipy.channels import dephasing_channel


def completely_dephasing_channel(d):

    '''
    Generates the completely dephasing channel in d dimensions. This channel
    eliminates the off-diagonal elements (in the standard basis) of the input operator.
    '''
    
    if d==2:
        p=1/2
        return dephasing_channel(p,d=d)[0]
    else:
        p=(1/d)*np.ones(d)
        return dephasing_channel(p,d=d)