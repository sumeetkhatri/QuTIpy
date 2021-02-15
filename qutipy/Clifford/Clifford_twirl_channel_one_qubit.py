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

from qutipy.gates import Rx_i,Ry_i,Rz_i
from qutipy.channels import apply_channel
from qutipy.general_functions import dag,eye


def Clifford_twirl_channel_one_qubit(K,rho,sys=1,dim=[2]):

    '''
    Twirls the given channel with Kraus operators in K by the one-qubit 
    Clifford group on the given subsystem (specified by sys).
    '''

    n=int(np.log2(np.sum([d for d in dim])))

    C1=eye(2**n)                                                                                                                                          
    C2=Rx_i(sys,np.pi,n)                                                                                                                                 
    C3=Rx_i(sys,np.pi/2.,n)                                                                                                                              
    C4=Rx_i(sys,-np.pi/2.,n)                                                                                                                             
    C5=Rz_i(sys,np.pi,n)                                                                                                                                 
    C6=Rx_i(sys,np.pi,n)*Rz_i(sys,np.pi,n)                                                                                                                 
    C7=Rx_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi,n)                                                                                                              
    C6=Rx_i(sys,np.pi,n)*Rz_i(sys,np.pi,n)                                                                                                                 
    C8=Rx_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi,n)
    C9=Rz_i(sys,np.pi/2.,n)                                                                                                                              
    C10=Ry_i(sys,np.pi,n)*Rz_i(sys,np.pi/2.,n)                                                                                                             
    C11=Ry_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                                         
    C12=Ry_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                                          
    C13=Rz_i(sys,-np.pi/2.,n)                                                                                                                            
    C14=Ry_i(sys,np.pi,n)*Rz_i(sys,-np.pi/2.,n)                                                                                                            
    C15=Ry_i(sys,-np.pi/2.,n)*Rz_i(sys,-np.pi/2.,n)                                                                                                        
    C16=Ry_i(sys,np.pi/2.,n)*Rz_i(sys,-np.pi/2.,n)                                                                                                         
    C17=Rz_i(sys,-np.pi/2.,n)*Rx_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)
    C18=Rz_i(sys,np.pi/2.,n)*Rx_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                       
    C19=Rz_i(sys,np.pi,n)*Rx_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                          
    C20=Rx_i(sys,np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                                          
    C21=Rz_i(sys,np.pi/2.,n)*Rx_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                      
    C22=Rz_i(sys,-np.pi/2.,n)*Rx_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                     
    C23=Rx_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)                                                                                                         
    C24=Rx_i(sys,np.pi,n)*Rx_i(sys,-np.pi/2.,n)*Rz_i(sys,np.pi/2.,n)

    C=[C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24]

    rho_twirl=0

    for i in range(len(C)):
        rho_twirl+=(1./24.)*C[i]@apply_channel(K,dag(C[i])@rho@C[i],sys,dim)@dag(C[i])

    return rho_twirl,C