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
from scipy.linalg import expm

from qutipy.general_functions import tensor,syspermute,eye


def Ry_i(i,t,n):

	'''
	Rotation about the Y axis on qubit i by angle t. The total number of
	qubits is n.
	'''

	dims=2*np.ones(n)
	dims=dims.astype(int)
	indices=np.linspace(1,n,n)
	indices_diff=np.setdiff1d(indices,i)
	perm_arrange=np.append(np.array([i]),indices_diff)
	perm_rearrange=np.zeros(n)

	for i in range(n):
		perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1
	
	perm_rearrange=perm_rearrange.astype(int)
	Sy=np.matrix([[0,-1j],[1j,0]])
	Ry=expm(-1j*t*Sy/2)
	out_temp=tensor(Ry,[eye(2),n-1])
	out=syspermute(out_temp,perm_rearrange,dims)

	return np.matrix(out)