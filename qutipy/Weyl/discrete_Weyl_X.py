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


from qutipy.general_functions import ket,dag



def discrete_Weyl_X(d):

    '''
    Generates the X shift operators.
    '''

    X=ket(d,1)@dag(ket(d,0))

    for i in range(1,d):
        X=X+ket(d,(i+1)%d)@dag(ket(d,i))

    return X