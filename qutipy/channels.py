'''
This code is part of QuTIPy.

(c) Copyright Sumeet Khatri, 2020

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.

'''


from .base import *


def Pauli_channel(px,py,pz):

    '''
    Generates the Kraus operators, an isometric extension, and a unitary
    extension of the Pauli channel specified by the parameters px, py, pz.
    '''

    pI=1-px-py-pz

    Sx=np.matrix([[0,1],[1,0]])
    Sy=np.matrix([[0,-1j],[1j,0]])
    Sz=np.matrix([[1,0],[0,-1]])

    K=[np.sqrt(pI)*eye(2),np.sqrt(px)*Sx,np.sqrt(py)*Sy,np.sqrt(pz)*Sz]

    V,U=generate_channel_isometry(K,2,2)

    return K,V,U


def Pauli_channel_nQubit(n,p,alt_repr=False):

    '''
    Generates the Kraus operators, an isometric extension, and a unitary
    extension of the n-qubit Pauli channel specified by the 2^(2*n) parameters in
    p, which must be probabilities in order for the map to be a channel. (i.e.,
    they must be non-negative and sum to one.)

    If alt_repr=True, then the channel is of the form

    P(rho)=\sum_{a,b} p_{a,b} X^aZ^b(rho)Z^bX^a

    where a and b are n-bit strings
    (using the n-qubit X and Z operators as generated by the functions
    generate_nQubit_Pauli_X and generate_nQubit_Pauli_Z).
    '''

    K=[]

    if not alt_repr:
        S=list(itertools.product(*[range(0,4)]*n))
        for i in range(2**(2*n)):
            K.append(np.sqrt(p[i])*generate_nQubit_Pauli(list(S[i])))

        V,U=generate_channel_isometry(K,2**n,2**n)

        return K,V,U

    else:  #alt_repr==True
        S=list(itertools.product(*[range(0,2)]*n))
        count=0
        for a in S:
            a=list(a)
            for b in S:
                b=list(b)
                K.append(np.sqrt(p[count])*generate_nQubit_Pauli_X(a)*generate_nQubit_Pauli_Z(b))
                count+=1

        V,U=generate_channel_isometry(K,2**n,2**n)

        return K,V,U


def Pauli_channel_coeffs(K,n,as_dict=False):

    '''
    Generates the coefficients c_{a,b} such that

        P(X^aZ^b)=c_{a,b}X^aZ^b,
    
    for the Pauli channel P with the Kraus operators in K.
    '''

    if as_dict:
        c={}
    else:
        c=[]

    S=list(itertools.product(*[range(0,2)]*n))
    #print(S)

    for a in S:
        for b in S:
            Xa=generate_nQubit_Pauli_X(list(a))
            Zb=generate_nQubit_Pauli_Z(list(b))
            if as_dict:
                c[(a,b)]=(1/2**n)*np.trace((Xa*Zb).H*apply_channel(K,Xa*Zb))
            else:
                c.append((1/2**n)*np.trace((Xa*Zb).H*apply_channel(K,Xa*Zb)))

    return c


def depolarizing_channel_nQubits(n,p):

    '''
    For 0<=p<=1, this returns the n-qubit Pauli channel given by
    p[0]=1-p, p[i]=p/(2^(2*n)-1) for all i>=1.
    '''

    p=[1-p]+[p/(2**(2*n)-1) for i in range(2**(2*n)-1)]

    return Pauli_channel_nQubit(n,p,alt_repr=True)


def depolarizing_channel(p):

    '''
    For 0<=p<=1, this returns the one-qubit Pauli channel given by px=py=pz=p/3.
    '''

    return Pauli_channel(p/3.,p/3.,p/3.)


def depolarizing_channel_n_uses(p,n,rho,m):


    '''
    Generates the output state corresponding to the depolarizing channel
    applied to each one of n systems in the joint state rho. p is the 
    depolarizing probability as defined in the function "depolarizing_channel"
    above.

    If rho contains m>n systems, then the first m-n systems are left alone.
    '''

    dims=2*np.ones(m).astype(int)

    rho_out=np.zeros((2**m,2**m))

    for k in range(n+1):
        indices=list(itertools.combinations(range(1,n+1),k))

        #print k,indices

        for index in indices:
            index=list(index)

            index=np.array(index)+(m-n)
            index=list(index.astype(int))

            index_diff=np.setdiff1d(range(1,m+1),index)

            perm_arrange=np.append(index,index_diff).astype(int)
            perm_rearrange=np.zeros(m)

            for i in range(m):
                perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1

            perm_rearrange=perm_rearrange.astype(int)

            mix=eye(2**k)/2.**k

            rho_part=TrX(rho,index,dims)

            rho_out=rho_out+(4*p/3.)**k*(1-(4*p/3.))**(n-k)*syspermute(tensor(mix,rho_part),perm_rearrange,dims)

    return np.matrix(rho_out)


def bit_flip_channel(p):

    '''
    Generates the channel rho -> (1-p)*rho+p*X*rho*X. 
    '''

    return Pauli_channel(p,0,0)


def dephasing_channel(p,d=2):

    '''
    Generates the channel rho -> (1-p)*rho+p*Z*rho*Z. (In the case d=2.)

    For d>=2, we let p be a list of d probabilities, and we use the su(d) generators
    (specifically, the last d-1 of them) to define the channel.

    For p=1/d, we get the completely dephasing channel.
    '''
    if d==2:
        return Pauli_channel(0,0,p)
    else:
        S=su_generators(d)[d**2-(d-1):d**2]
        K=[]
        K.append(np.sqrt(p[0]*eye(d)))
        for k in range(1,d):
            K.append(np.sqrt(p[k])*S[k-1])
        
        return K


def completely_dephasing_channel(d):

    '''
    Generates the completely dephasing channel in d dimensions. This channels
    eliminates the off-diagonal elements (in the standard basis) of the input operator.
    '''
    
    if d==2:
        p=1/2
        return dephasing_channel(p,d=d)[0]
    else:
        p=(1/d)*np.ones(d)
        return dephasing_channel(p,d=d)


def phase_damping_channel(p):

    '''
    Generates the phase damping channel.
    '''

    K1=np.matrix([[1,0],[0,np.sqrt(p)]])
    K2=np.matrix([[0,0],[0,np.sqrt(1-p)]])

    return [K1,K2]


def BB84_channel(Q):

    '''
    Generates the channel corresponding to the BB84 protocol with
    equal X and Z errors, given by the QBER Q. The definition of this
    channel can be found in:

        "Additive extensions of a quantum channel", by
            Graeme Smith and John Smolin. (arXiv:0712.2471)

    '''

    return Pauli_channel(Q-Q**2,Q**2,Q-Q**2)



def amplitude_damping_channel(gamma):

    '''
    Generates the amplitude damping channel.
    '''

    A1=np.matrix([[1,0],[0,np.sqrt(1-gamma)]])
    A2=np.matrix([[0,np.sqrt(gamma)],[0,0]])

    return [A1,A2]


def generalized_amplitude_damping_channel(gamma,N):

    '''
    Generates the generalized amplitude damping channel.
    '''

    if N==0:
        return amplitude_damping_channel(gamma)
    elif N==1:
        A1=np.matrix([[np.sqrt(1-gamma),0],[0,1]])
        A2=np.matrix([[0,0],[np.sqrt(gamma),0]])
        return [A1,A2]
    else:
        A1=np.sqrt(1-N)*np.matrix([[1,0],[0,np.sqrt(1-gamma)]])
        A2=np.sqrt(1-N)*np.matrix([[0,np.sqrt(gamma)],[0,0]])
        A3=np.sqrt(N)*np.matrix([[np.sqrt(1-gamma),0],[0,1]])
        A4=np.sqrt(N)*np.matrix([[0,0],[np.sqrt(gamma),0]])

        return [A1,A2,A3,A4]


def compose_channels(C):

    '''
    Takes a composition of channels. The variable C should be a list of lists,
    with each list consisting of the Kraus operators of the channels to be composed.

    If C=[K1,K2,...,Kn], then this function returns the composition such that
    the channel corresponding to K1 is applied first, then K2, etc.
    '''

    lengths=[]
    for c in C:
        lengths.append(len(c))
    
    combs=list(itertools.product(*[range(length) for length in lengths]))

    K_n=[]

    for comb in combs:
        tmp=1
        for i in range(len(comb)):
            tmp=C[i][comb[i]]*tmp
        K_n.append(tmp)

    return K_n


def apply_channel(K,rho,sys=None,dim=None,adjoint=False):

    '''
    Applies the channel with Kraus operators in K to the state rho on
    systems specified by sys. The dimensions of the subsystems on which rho
    acts are given by dim.

    If adjoint is True, then this function applies the adjoint of the given
    channel.
    '''

    if adjoint==True:
        K_tmp=K
        K=[]
        K=[K_tmp[i].H for i in range(len(K_tmp))]

    if sys==None:
        #if type(rho)==cvx.expressions.variable.Variable:
        #    return np.sum([K[i]*rho*K[i].H for i in range(len(K))],0)
        #else:
        return np.matrix(np.sum([K[i]*rho*K[i].H for i in range(len(K))],0))
    else:
        A=[]
        for i in range(len(K)):
            X=1
            for j in range(len(dim)):
                if j+1==sys:
                    X=tensor(X,K[i])
                else:
                    X=tensor(X,eye(dim[j]))
            A.append(X)
        #if type(rho)==cvx.expressions.variable.Variable:
        #    return np.sum([A[i]*rho*A[i].H for i in range(len(A))],0)
        #else:
        return np.matrix(np.sum([A[i]*rho*A[i].H for i in range(len(A))],0))



def Choi_representation(K,dimA):

    '''
    Calculates the Choi representation of the map with Kraus operators K.
    dimA is the dimension of the input space of the channel.

    The Choi represenatation is defined with the channel acting on the second
    half of the maximally entangled vector.
    '''


    #Gamma=np.sqrt(dimA)*MaxEnt_state(dimA)
    Gamma=MaxEnt_state(dimA,normalized=False)

    return np.matrix(apply_channel(K,Gamma*Gamma.H,2,[dimA,dimA]),dtype=np.complex)


def Natural_representation(K):

    '''
    Calculates the natural representation of the channel (in the standard basis)
    given by the Kraus operators in K. In terms of the Kraus operators, the natural
    representation of the channel in the standard basis is given by

    N=sum_i K_i ⊗ conj(C_i),

    where the sum is over the Kraus operators K_i in K.
    '''

    return np.matrix(np.sum([tensor(k,np.conjugate(k)) for k in K],1))



def Kraus_representation(P,dimA,dimB):

    '''
    Takes a Choi representation P of a channel and returns its Kraus representation.
    
    The Choi representation is defined with the channel acting on the second half of
    the maximally entangled vector.
    '''

    D,U=eig(P)

    U_cols=U.shape[1]

    U=np.matrix(U)

    #print(U)

    # Need to check if the matrix U generated by eig is unitary (up to numerical precision)
    check1=np.allclose(eye(dimA*dimB),U*U.H)
    check2=np.allclose(eye(dimA*dimB),U.H*U)

    if check1 and check2:
        U=np.array(U)
    
    # If U is not unitary, use Gram-Schmidt to make it unitary (i.e., make the columns of U orthonormal)
    else:
        C=gram_schmidt([U[:,i] for i in range(U_cols)],dimA*dimB)
        U=np.sum([tensor(ket(U_cols,i).H,C[i]) for i in range(U_cols)],0)
        #print(U)
    K=[]

    for i in range(U_cols):
        Col=U[:,i]
        K_tmp=np.matrix(np.sqrt(D[i])*Col.reshape([dimA,dimB]))
        K.append(K_tmp.transpose())

    return K


def Choi_to_Natural(C_AB,dimA,dimB):

    '''
    Takes the Choi representation of a map and outputs its natural representation.

    The Choi represenatation Q of the channel acts as:
    
        vec(N(rho))=Q*vec(rho),
    
    where N is the channel in question. It can be obtained from the Choi representation
    with a simple reshuffling of indices.
    '''

    C_AB=np.array(C_AB)

    return np.matrix(np.reshape(C_AB,[dimA,dimB,dimA,dimB]).transpose((0,2,1,3)).reshape([dimA*dimA,dimB*dimB])).T


def n_channel_uses(K,n):

    '''
    Given the Kraus operators K of a channel, this function generates the
    Kraus operators corresponding to the n-fold tensor power of the channel.
    dimA is the dimension of the input space, and dimB the dimension of the
    output space.
    '''

    r=len(K)   # Number of Kraus operators

    combs=list(itertools.product(*[range(r)]*n))

    K_n=[]

    for comb in combs:
        #print comb
        tmp=1
        for i in range(n):
            tmp=tensor(tmp,K[comb[i]])
        K_n.append(tmp)

    return K_n


def tensor_channels(C):

    '''
    Takes the tensor product of the channels in C.

    C is a set of sets of Kraus operators.
    '''

    lengths=[]
    for c in C:
        lengths.append(len(c))
    
    combs=list(itertools.product(*[range(length) for length in lengths]))

    K_n=[]

    for comb in combs:
        tmp=1
        for i in range(len(comb)):
            tmp=tensor(tmp,C[i][comb[i]])
        K_n.append(tmp)

    return K_n



def channel_scalar_multiply(K,x):

    '''
    Multiplies the channel with Kraus operators in K by the scalar x.
    This means that each Kraus operator is multiplied by sqrt(x)!
    '''

    K_new=[]
    
    for i in range(len(K)):
        K_new.append(np.sqrt(x)*K[i])

    return K_new


def generate_channel_isometry(K,dimA,dimB):

    '''
    Generates an isometric extension of the
    channel specified by the Kraus operators K. dimA is the dimension of the
    input space of the channel, and dimB is the dimension of the output space
    of the channel. If dimA=dimB, then the function also outputs a unitary
    extension of the channel given by a particular construction.
    '''

    dimE=len(K)

    V=np.matrix(np.sum([tensor(K[i],ket(dimE,i)) for i in range(dimE)],0))

    if dimA==dimB:
        # In this case, the unitary we generate has dimensions dimA*dimE x dimA*dimE
        U=tensor(V,ket(dimE,0).H)
        states=[V*ket(dimA,i) for i in range(dimA)]
        for i in range(dimA*dimE-dimA):
            states.append(RandomPureState(dimA*dimE))

        states_new=gram_schmidt(states,dimA*dimE)

        count=dimA
        for i in range(dimA):
            for j in range(1,dimE):
                U=U+tensor(states_new[count],ket(dimA,i).H,ket(dimE,j).H)
                count+=1
        
        return V,U
    else:
        return V


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
        rho_twirl+=(1./24.)*C[i]*apply_channel(K,C[i].H*rho*C[i],sys,dim)*C[i].H

    return rho_twirl,C


def diamond_norm(J,dimA,dimB,display=True):

    '''
    Computes the diamond norm of a superoperator with Choi representation J.
    dimA is the dimension of the input space of the channel, and dimB is the
    dimension of the output space.

    The form of the SDP used comes from Theorem 3.1 of:
        
        'Simpler semidefinite programs for completely bounded norms',
            Chicago Journal of Theoretical Computer Science 2013,
            by John Watrous
    '''

    '''
    The Choi representation J in the above paper is defined using a different
    convention:
        J=(N\otimes I)(|Phi^+><Phi^+|).
    In other words, the channel N acts on the first half of the maximally-
    entangled state, while the convention used throughout this code stack
    is
        J=(I\otimes N)(|Phi^+><Phi^+|).
    We thus use syspermute to convert to the form used in the aforementioned
    paper.
    '''

    J=np.matrix(syspermute(J,[2,1],[dimA,dimB]))

    X=cvx.Variable((dimA*dimB,dimA*dimB))
    rho0=cvx.Variable((dimA,dimA),PSD=True)
    rho1=cvx.Variable((dimA,dimA),PSD=True)

    M=cvx.kron(ket(2,0)*ket(2,0).H,cvx.kron(eye(dimB),rho0))+cvx.kron(ket(2,0)*ket(2,1).H,X)+cvx.kron(ket(2,1)*ket(2,0).H,X.H)+cvx.kron(ket(2,1)*ket(2,1).H,cvx.kron(eye(dimB),rho1))

    c=[]
    c+=[M>>0,cvx.trace(rho0)==1,cvx.trace(rho1)==1]

    obj=cvx.Maximize((1./2.)*cvx.real(cvx.trace(J.H*X))+(1./2.)*cvx.real(cvx.trace(J*X.H)))

    prob=cvx.Problem(obj,constraints=c)

    prob.solve(solver=cvx.CVXOPT,verbose=display,eps=1e-8)
    #prob.solve(verbose=display)

    return prob.value


def ent_fidelity_channel(K,d):

    '''
    Finds the entanglement fidelity of the channel given by the set K of 
    Kraus operators. d is the dimension of the input space.
    '''

    Bell=MaxEnt_state(d)

    K_choi=(1./d)*Choi_representation(K,d)

    return np.real(np.trace((Bell*Bell.H)*K_choi))


def avg_fidelity(K,dimA):

    '''
    Calculates the average fidelity of a channel using its entanglement
    fidelity with respect to the maximally mixed state (see, e.g., Eq.
    (9.245) of Wilde's book.)

    K is the set of Kraus operators of the channel, and dimA is the dimension
    of the input space of the channel.
    '''

    choi_state=(1./dimA)*Choi_representation(K,dimA)

    return (dimA*ent_fidelity(choi_state,dimA)+1)/(dimA+1)


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

        F+=np.real(np.trace((state*state.H)*apply_channel(K,state*state.H)))

    return (1./6.)*F


def entanglement_distillation(rho1,rho2,outcome=1,twirl_after=False):

    '''
    Applies a particular entanglement distillation channel to the two two-qubit states
    rho1 and rho2. [cite]

    The channel is probabilistic. If the variable outcome=1, then the function returns
    the two-qubit state conditioned on the success of the distillation protocol.
    '''

    Sx=np.matrix([[0,1],[1,0]])
    CNOT=CNOT_ij(1,2,2)
    proj1=ket(2,1)*ket(2,1).H

    P1=tensor(eye(2),proj1,eye(2),proj1)
    P0=eye(16)-P1
    C=tensor(CNOT,CNOT)
    K1=P1*C
    K0=P0*C

    rho_in=syspermute(tensor(rho1,rho2),[1,3,2,4],[2,2,2,2]) # rho_in==rho_{A1A2B1B2}

    if outcome==1:
        # rho_out is unnormalized. The trace of rho_out is equal to the success probability.
        rho_out=TrX(K1*rho_in*K1.H,[2,4],[2,2,2,2])
        if twirl_after:
            rho_out=isotropic_twirl_state(rho_out,2)

    elif outcome==0:
        # rho_out is unnormalized. The trace of rho_out is equal to the failure probability.
        rho_out=TrX(K0*rho_in*K0.H,[2,4],[2,2,2,2])

    return rho_out