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


import numpy as np
from numpy.linalg import norm, qr
from scipy.linalg import sqrtm,eig,fractional_matrix_power,logm,expm
from scipy.stats import unitary_group
from scipy.optimize import minimize
import itertools
#from optimize_unitarySD import unitarySD_selftuning,unitarySD_proj 
import cvxpy as cvx



####################################################################
## BASIC OPERATIONS
####################################################################


def ket(dim,*args):

    '''
    Generates a standard basis vector in dimension dim.

    For example, ket(2,0)=|0>=[1,0] and ket(2,1)=|1>=[0,1].

    In general, ket(d,j), for j between 0 and d-1, generates a column vector
    (as a numpy matrix) in which the jth element is equal to 1 and the rest
    are equal to zero.

    ket(d,[j_1,j_2,...,j_n]) generates the tensor product |j_1>|j_2>...|j_n> of
    d-dimensional basis vectors.
    '''

    args=np.array(args)

    if args.size==1:
        num=args[0]
        out=np.zeros([dim,1])
        out[num]=1
    else:
        args=args[0]
        out=ket(dim,args[0])
        for j in range(1,len(args)):
            out=np.kron(out,ket(dim,args[j]))
    
    return np.matrix(out)



def tensor(*args):

    '''
    Takes the tensor product of an arbitrary number of matrices/vectors.
    '''

    M=1

    for j in range(len(args)):
        if isinstance(args[j],list):
            for k in range(args[j][1]):
                M=np.kron(M,args[j][0])
        else:
            M=np.kron(M,args[j])

    return np.matrix(M)


def Tr(A):

    '''
    Takes the trace of the matrix A. The object A should be a numpy matrix.
    '''

    if type(A)!=np.matrix:
        A=np.matrix(A)

    return np.trace(A)
    

def TrX(state,sys,dim):

    '''
    sys is a list of systems over which to take the partial trace (i.e., the
    systems to discard).

    Example: If rho_AB is a bipartite state with dimA the dimension of system A 
    and dimB the dimension of system B, then

    TrX(rho_AB,[2],[dimA,dimB]) gives the density matrix on

    system A, i.e., rho_A:=TrX[rho_AB].

    Similarly, TrX(rho_AB,[1],[dimA,dimB]) discards the first subsystem,
    returning the density matrix of system B.

    If rho_ABC is a tripartite state, then, e.g.,

    TrX(rho_ABC,[1,3],[dimA,dimB,dimC])

    discards the first and third subsystems, so that we obtain the density
    matrix for system B.

    '''

    if not sys:  # If sys is empty, just return the original state
        return state
    elif len(sys)==len(dim):  # If tracing over all systems
        return np.trace(state)
    else:

        if state.shape[1]==1:
            state=state*state.H

        num_sys=len(dim)
        total_sys=range(1,num_sys+1)

        dims_sys=[dim[s-1] for s in sys] # Dimensions of the system to be traced over
        dims_keep=[dim[s-1] for s in list(set(total_sys)-set(sys))]
        dim_sys=np.product(dims_sys)
        dim_keep=np.product(dims_keep)

        perm=sys+list(set(total_sys)-set(sys))
        state=syspermute(state,perm,dim)

        state=np.array(state)
        dim=[dim_sys]+dims_keep
        state_reshape=np.reshape(state,dim+dim)
        state_reshape=np.sum(np.diagonal(state_reshape,axis1=0,axis2=len(dim)),axis=-1)
        state=np.reshape(state_reshape,(dim_keep,dim_keep))

        return np.matrix(state)


def Tx(state,sys,dim):

    '''
    Takes the partial transpose on systems given by sys. dim is a list of 
    the dimensions of each of the subsystems in state.

    Example: If rho_AB is a bipartite state with dimA the dimension of system A 
    and dimB the dimension of system B, then

    Tx(rho_AB,[2],[dimA,dimB]) takes the transpose on system B.

    Similarly, Tx(rho_AB,[1],[dimA,dimB]) takes the transpose on system A.

    If rho_ABC is a tripartite state, then, e.g.,

    TrX(rho_ABC,[1,3],[dimA,dimB,dimC])

    takes the transpose on systems A and C.

    This also works for non-square matrices provided dim is a list of tuples,
    with the first element of the tuple specifying the dimension of the rows
    of each subsystem and the second element of the tuple specifying the 
    dimension of the columns of each subsystem; e.g., dim=[(2,4),(3,5)] means
    that the first tensor factor lives in an operator space taking a 
    four-dimensional space to a two-dimensional space, and the second tensor
    factor lives in an operator space taking a 5-dimensional space to a 
    3-dimensional space.
    '''

    if state.shape[1]==1:
        state=state*state.H

    state=np.array(state)

    n=len(dim)  # Number of subsystems in the state

    if type(dim[0])==tuple or type(dim[0])==list:  # When state is a non-square matrix
        dim_row=[dim[i][0] for i in range(n)]
        dim_col=[dim[i][1] for i in range(n)]
    elif type(dim[0])==int:  # When state is a square matrix
        dim_row=dim
        dim_col=dim

    state_reshape=np.reshape(state,dim_row+dim_col)

    axes=list(range(2*n))

    for i in range(len(sys)):
        axes[sys[i]-1],axes[n+sys[i]-1]=axes[n+sys[i]-1],axes[sys[i]-1]
        if type(dim[0])==tuple or type(dim[0])==list:
            dim[sys[i]-1]=list(dim[sys[i]-1])
            dim[sys[i]-1][0],dim[sys[i]-1][1]=dim[sys[i]-1][1],dim[sys[i]-1][0]
            #dim[sys[i]-1]=tuple(dim[sys[i]-1])
        else:
            continue

    state_reshape=np.transpose(state_reshape,tuple(axes))

    if type(dim[0])==tuple or type(dim[0])==list:
        dim_row=[dim[i][0] for i in range(n)]
        dim_col=[dim[i][1] for i in range(n)]
        dim_total=(np.product(dim_row),np.product(dim_col))
    elif type(dim[0])==int:
        dim_row=dim
        dim_col=dim
        dim_total=(np.product(dim),np.product(dim))

    state_new=np.reshape(state_reshape,dim_total)

    return np.matrix(state_new)


def syspermute(p,perm,dim):

    '''
    Permutes order of subsystems in the multipartite operator p.
    
    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    subsystems.
    '''

    # If p is defined using np.matrix(), then it must first be converted
    # to a numpy array, or else the reshaping below won't work.
    p=np.array(p) 


    n=len(dim)
    d=p.shape

    perm=np.array(perm)
    dim=np.array(dim)

    if d[0]==1 or d[1]==1:
        # For a pure state
        perm=perm-1
        tmp=np.reshape(p,dim)
        q=np.reshape(np.transpose(tmp,perm),d)
    elif d[0]==d[1]:
        # For a mixed state (density matrix)
        perm=perm-1
        perm=np.append(perm,n+perm)
        dim=np.append(dim,dim)
        tmp=np.reshape(p,dim)
        q=np.reshape(np.transpose(tmp,perm),d)

    return np.matrix(q)


def get_subblock(X,sys,indices,dim):

    '''
    For the multipartite operator X, with local dimensions given by the list dim,
    this function extracts the subblock such that the systems in sys have the
    values given by those in indices. indices should be a list of tuples, each
    tuple corresponding to a system in sys. For each tuple, the first element
    gives the row and the second the column.

    For example, the subblock (<i|\otimes 1\otimes <j|)X(|k>\otimes 1\otimes |l>)
    is given by get_subblock(X,[1,3],[(i,k),(j,l)],[dim1,dim2,dim3]), so the result
    is a dim2 x dim2 matrix, which is the desired subblock.
    '''

    X=np.array(X)

    X_reshape=np.reshape(X,dim+dim)

    num_sys=len(dim)
    total_sys=range(1,num_sys+1)
    dims_keep=[dim[s-1] for s in list(set(total_sys)-set(sys))]

    to_slice=[slice(None) for i in range(2*num_sys)]

    count=0
    for s in total_sys:
        if s in sys:
            to_slice[s-1]=indices[count][0]
            to_slice[num_sys+s-1]=indices[count][1]
            count+=1
    
    X_reshape=X_reshape[to_slice]

    X_reshape=np.reshape(X_reshape,dims_keep+dims_keep)

    return X_reshape




####################################################################
## BASIC STATES AND GATES
####################################################################


def eye(n):

    '''
    Generates the nxn identity matrix.
    '''

    return np.matrix(np.identity(n,dtype=int))


def MaxEnt_state(dim,normalized=True):

    '''
    Generates the dim-dimensional maximally entangled state, which is defined as

    (1/sqrt(dim))*(|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If normalized=False, then the function returns the unnormalized maximally entangled
    vector.
    '''

    if normalized:
        return (1./np.sqrt(dim))*np.matrix(np.sum([ket(dim,[i,i]) for i in range(dim)],0))
    else:
        return np.matrix(np.sum([ket(dim,[i,i]) for i in range(dim)],0))
   

def MaxMix_state(dim):

    '''
    Generates the dim-dimensional maximally mixed state.
    '''

    return eye(dim)/dim


def isotropic_state(p,d,fidelity=False):

    '''
    Generates the isotropic state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_Iso = p*|Bell><Bell|+(1-p)*eye(d^2)/d^2,

    where -1/(d^2-1)<=p<=1.

    If fidelity=True, then the function returns a different parameterization of
    the isotropic state in which the parameter p is the fidelity of the state
    with respect to the maximally entangled state.
    '''

    Bell=MaxEnt_state(d)

    if fidelity:
        return p*Bell*Bell.H+((1-p)/3)*(eye(d**2)-Bell*Bell.H)
    else:
        return p*Bell*Bell.H+(1-p)*eye(d**2)/d**2


def Werner_state(p,d):

    '''
    Generates the Werner state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_W=(1/(d^2-dp))*(eye(d^2)-p*SWAP),

    where SWAP is the swap operator between two d-dimensional systems and 
    p is between -1 and 1.
    '''

    return (1/(d**2-d*p))*(eye(d**2)-p*SWAP([1,2],[d,d]))


def SWAP(sys,dim):

    '''
    Generates a swap matrix between the pair of systems in sys. dim is a list
    of the dimensions of the subsystems.

    For example, SWAP([1,2],[2,2]) generates the two-qubit swap matrix.
    '''

    dim_total=np.product(dim)

    n=len(dim)
    sys_rest=list(np.setdiff1d(range(1,n+1),sys))
    perm=sys+sys_rest
    p={}

    for i in range(1,n+1):
        p[i]=perm[i-1]

    p2={v:k for k,v in p.items()}

    perm_rearrange=list(p2.values())

    dim1=dim[sys[0]-1] # Dimension of the first subsystem to be swapped
    dim2=dim[sys[1]-1] # Dimension of the second subsystem to be swapped

    dim_rest=int(float(dim_total)/float(dim1*dim2))

    G1=np.matrix(np.sum([ket(dim1,[i,i]) for i in range(dim1)],0))
    G2=np.matrix(np.sum([ket(dim2,[i,i]) for i in range(dim2)],0))

    G=G1*G2.H

    S=Tx(G,[2],[(dim1,dim2),(dim1,dim2)])

    P=tensor(S,eye(dim_rest))

    p_alt=list(np.array(list(p.values()))-1)

    P=syspermute(P,perm_rearrange,list(np.array(dim)[p_alt]))

    return P


def CNOT_ij(i,j,n):

	'''
	CNOT gate on qubits i and j, i being the control and j being the target.
	The total number of qubits is n.
	'''
	
	dims=2*np.ones(n)
	dims=dims.astype(int)
	
	indices=np.linspace(1,n,n)
	indices_diff=np.setdiff1d(indices,[i,j])
	
	perm_arrange=np.append(np.array([i,j]),indices_diff)
	perm_rearrange=np.zeros(n)

	for i in range(n):
		perm_rearrange[i]=np.argwhere(perm_arrange==i+1)[0][0]+1
	
	perm_rearrange=perm_rearrange.astype(int)

	Sx=np.matrix([[0,1],[1,0]])
	CX=tensor(ket(2,0)*np.transpose(ket(2,0)),eye(2))+tensor(ket(2,1)*np.transpose(ket(2,1)),Sx)

	out_temp=tensor(CX,[eye(2),n-2])

	out=syspermute(out_temp,perm_rearrange,dims)

	return out


def Rx(t):

    '''
    Generates the unitary rotation matrix about the X axis on the Bloch sphere.
    '''

    Sx=np.matrix([[0,1],[1,0]])
    return np.matrix(expm(-1j*t*Sx/2.))



def Rx_i(i,t,n):

	'''
	Rotation about the X axis on qubit i by angle t. The total number of
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
	Sx=np.matrix([[0,1],[1,0]])
	Rx=expm(-1j*t*Sx/2)
	out_temp=tensor(Rx,[eye(2),n-1])
	out=syspermute(out_temp,perm_rearrange,dims)

	return np.matrix(out)


def Ry(t):
    
    '''
    Generates the unitary rotation matrix about the Y axis on the Bloch sphere.
    '''

    Sy=np.matrix([[0,-1j],[1j,0]])
    return np.matrix(expm(-1j*t*Sy/2.))


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


def Rz(t):

    '''
    Generates the unitary rotation matrix about the Z axis on the Bloch sphere.
    '''

    Sz=np.matrix([[1,0],[0,-1]])
    return np.matrix(expm(-1j*t*Sz/2.))


def Rz_i(i,t,n):

	'''
	Rotation about the Z axis on qubit i by angle t. The total number of
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
	Sz=np.matrix([[1,0],[0,-1]])
	Rz=expm(-1j*t*Sz/2)
	out_temp=tensor(Rz,[eye(2),n-1])
	out=syspermute(out_temp,perm_rearrange,dims)

	return np.matrix(out)


def H_i(i,n):

    '''
    Generates the matrix for the Hadamard gate applied to the ith qubit.
    n is the total number of qubits.
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
    H=(1/np.sqrt(2))*np.matrix([[1,1],[1,-1]])
    out_temp=tensor(H,[eye(2),n-1])
    out=syspermute(out_temp,perm_rearrange,dims)

    return np.matrix(out)


def S_i(i,n):

    '''
    Generates the matrix for the S gate applied to the ith qubit.
    n is the total number of qubits. The S gate is defined as:

        S:=[[1 0],
            [0 1j]]

    It is one of the generators of the Clifford group.
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
    S=np.matrix([[1,0],[0,1j]])
    out_temp=tensor(S,[eye(2),n-1])
    out=syspermute(out_temp,perm_rearrange,dims)

    return np.matrix(out)




####################################################################
## DISTANCE MEASURES AND NORMS
####################################################################

def unitary_distance(U,V):
    
    '''
    Checks whether two unitaries U and V are the same (taking into account global phase) by using the distance measure:
    
    1-(1/d)*|Tr[UV^\dagger]|,
    
    where d is the dimension of the space on which the unitaries act.
    
    U and V are the same if and only if this is equal to zero; otherwise, it is greater than zero.
    '''
    
    U=np.matrix(U)
    V=np.matrix(V)

    d=U.shape[0]
    
    return 1-(1/d)*np.abs(np.trace(U*V.H))


def trace_distance_pure_states(psi,phi):

    '''
    Computes the squared trace distance between two pure states psi and phi,
    i.e.,

    || |psi><psi|-|phi><phi| ||_1^2

    '''

    return 1-np.trace((psi*psi.H)*(phi*phi.H))


def trace_norm(X):

    '''
    Finds the trace norm of the matrix X. (Sum of the singular values.)
    '''

    return norm(X,ord='nuc')


def fidelity(rho,sigma):

    '''
    Returns the fidelity between the states rho and sigma.
    '''

    return trace_norm(np.matrix(sqrtm(rho))*np.matrix(sqrtm(sigma)))**2


def ent_fidelity(sigma,d):

    '''
    Finds the fidelity between the state sigma and the Bell state.
    d is the dimension.
    '''

    Bell=MaxEnt_state(d)

    return np.real(np.trace((Bell*Bell.H)*sigma))


def spectral_norm(X):

    '''
    Finds the spectral norm (also known as the operator norm) of the matrix X.
    (The largest singular value.)
    '''

    return norm(X,ord=2)




####################################################################
## RANDOM STATES AND UNITARIES
####################################################################


def RandomUnitary(dim):

    '''
    Generates a random unitary. 
    '''

    return np.matrix(unitary_group.rvs(dim))


def RandomDensityMatrix(dim,comp=True,*args):

    '''
    Generates a random density matrix.
    
    Optional argument is for the rank r of the state.
    
    Optional argument comp is for whether the state should have
    complex entries
    '''

    args=np.array(args)

    if args.size==0:
        r=dim
    else:
        r=args[0]

    if comp:
        gin=np.matrix(np.random.randn(dim,r)+1j*np.random.randn(dim,r))
        rho=gin*gin.H
    else:
        gin=np.matrix(np.random.randn(dim,r))
        rho=gin*gin.H

    return rho/np.trace(rho)


def RandomPureState(dim,rank=None):

    '''
    Generates a random pure state.

    For multipartite states, dim should be a list of dimensions for each
    subsystem. In this case, the rank variable is for the Schmidt rank. To specify
    the Schmidt rank, there has to be a bipartition of the systems, so that dim
    has only two elements.
    '''
    
    if rank==None:
        if type(dim)==list:
            dim=np.prod(dim)

        psi=np.matrix(np.random.randn(dim)).H+1j*np.matrix(np.random.randn(dim)).H

        psi=psi/norm(psi)

        return psi
    else:
        dimA=dim[0]
        dimB=dim[1]
        k=rank
        psi_k=MaxEnt_state(k)
        a=np.matrix(np.random.rand(dimA*k)).H+1j*np.matrix(np.random.rand(dimA*k)).H
        b=np.matrix(np.random.rand(dimB*k)).H+1j*np.matrix(np.random.rand(dimB*k)).H

        psi_init=syspermute(tensor(a,b),[1,3,2,4],[k,dimA,k,dimB])

        psi=tensor(psi_k.H,eye(dimA*dimB))*psi_init

        psi=psi/norm(psi)

        return psi



####################################################################
## PAULI MATRICES
####################################################################


def generate_nQubit_Pauli(indices):

    '''
    Generates a tensor product of Pauli operators for n qubits. indices is a list
    of indices i specifying the Pauli operator for each tensor factor. i=0 is the identity, i=1 is sigma_x,
    i=2 is sigma_y, and i=3 is sigma_z.
    '''

    Id=eye(2)
    Sx=np.matrix([[0,1],[1,0]])
    Sy=np.matrix([[0,-1j],[1j,0]])
    Sz=np.matrix([[1,0],[0,-1]])

    out=1

    for index in indices:
        if index==0:
            out=tensor(out,Id)
        elif index==1:
            out=tensor(out,Sx)
        elif index==2:
            out=tensor(out,Sy)
        elif index==3:
            out=tensor(out,Sz)
    
    return out


def generate_nQubit_Pauli_X(indices):

    '''
    Generates a tensor product of Pauli-X operators for n qubits. indices is
    a list of bits.
    '''

    Id=eye(2)
    Sx=np.matrix([[0,1],[1,0]])

    out=1

    for index in indices:
        if index==0:
            out=tensor(out,Id)
        elif index==1:
            out=tensor(out,Sx)
        else:
            return('Error: Indices must be bits, either 0 or 1!')
    
    return out


def generate_nQubit_Pauli_Z(indices):

    '''
    Generates a tensor product of Pauli-Z operators for n qubits. indices is
    a list of bits.
    '''

    Id=eye(2)
    Sz=np.matrix([[1,0],[0,-1]])

    out=1

    for index in indices:
        if index==0:
            out=tensor(out,Id)
        elif index==1:
            out=tensor(out,Sz)
        else:
            return('Error: Indices must be bits, either 0 or 1!')
    
    return out


def Pauli_coeff_to_matrix(coeffs,n):

    '''
    Takes the coefficients of a matrix in the n-qubit Pauli basis and outputs it
    as a matrix.

    coeffs should be specified as a one-dimensional list or array in standard
    lexicographical ordering.
    '''

    all_indices=list(itertools.product(*[range(0,4)]*n))

    out=0+0j

    for i in range(len(all_indices)):
        out+=(1./2.**n)*coeffs[i]*generate_nQubit_Pauli(all_indices[i])

    return out


def nQubit_Pauli_coeff(X,n):

    '''
    Generates the coefficients of the matrix X in the n-qubit Pauli basis.
    The coefficients c_{alpha} are such that

    X=(1/2^n)\sum_{alpha} c_alpha \sigma_alpha

    The coefficients are returned in lexicographical ordering.
    '''

    X=np.matrix(X)

    indices=list(itertools.product(*[range(0,4)]*n))

    C=[]

    for index in indices:
        sigma_i=generate_nQubit_Pauli(index)
        C.append(np.trace(X.H*sigma_i))

    return C



def nQubit_quadratures(n):

    '''
    Returns the list of n-qubit "quadrature" operators, which are defined as
    (for two qubits)

        S[0]=Sx \otimes Id
        S[1]=Sz \otimes Id
        S[2]=Id \otimes Sx
        S[3]=Id \otimes Sz

    In general, for n qubits:

        S[0]=Sx \otimes Id \otimes ... \otimes Id
        S[1]=Sz \otimes Id \otimes ... \otimes Id
        S[2]=Id \otimes Sx \otimes ... \otimes Id
        S[3]=Id \otimes Sz \otimes ... \otimes Id
        .
        .
        .
        S[2n-2]=Id \otimes Id \otimes ... \otimes Sx
        S[2n-1]=Id\otimes Id \otimes ... \otimes Sz
    '''

    S={}

    Sx=np.matrix([[0,1],[1,0]])
    Sz=np.matrix([[1,0],[0,-1]])

    count=0

    for i in range(1,2*n+1,2):
        v=list(np.array(ket(n,count).H,dtype=np.int).flatten())
        S[i]=generate_nQubit_Pauli_X(v)
        S[i+1]=generate_nQubit_Pauli_Z(v)
        count+=1

    return S


def nQubit_cov_matrix(X,n):

    '''
    Using the n-qubit quadrature operators, we define the n-qubit "covariance matrix"
    as follows:

    V_{i,j}=Tr[X*S_i*S_j]
    '''


    S=nQubit_quadratures(n)

    V=np.matrix(np.zeros((2*n,2*n)),dtype=np.complex128)
    #V=np.matrix(np.zeros((2*n,2*n)),dtype=object)

    for i in range(2*n):
        for j in range(2*n):
            V[i,j]=np.trace(X*S[i+1]*S[j+1])
    
    return V


def nQubit_mean_vector(X,n):

    '''
    Using the n-qubit quadrature operators, we define the n-qubit "mean vector" as
    follows:

        r_i=Tr[X*S_i]
    '''


    S=nQubit_quadratures(n)

    r=np.matrix(np.zeros((2*n,1)),dtype=np.complex128)
    #r=np.matrix(np.zeros((2*n,1)),dtype=object)

    for i in range(2*n):
        r[i,0]=np.trace(X*S[i+1])

    return r


####################################################################
## CLIFFORD GROUP
####################################################################


def Clifford_group_one_qubit():
    '''
    Returns the 24 one-qubit Clifford gates.
    '''

    C1=eye(2)                                                                                                                                          
    C2=Rx(np.pi)                                                                                                                                 
    C3=Rx(np.pi/2.)                                                                                                                              
    C4=Rx(-np.pi/2.)                                                                                                                             
    C5=Rz(np.pi)                                                                                                                                 
    C6=Rx(np.pi)*Rx(np.pi)                                                                                                                 
    C7=Rx(np.pi/2.)*Rz(np.pi)                                                                                                              
    C6=Rx(np.pi)*Rz(np.pi)                                                                                                                 
    C8=Rx(-np.pi/2.)*Rz(np.pi)
    C9=Rz(np.pi/2.)                                                                                                                              
    C10=Ry(np.pi)*Rz(np.pi/2.)                                                                                                             
    C11=Ry(-np.pi/2.)*Rz(np.pi/2.)                                                                                                         
    C12=Ry(np.pi/2.)*Rz(np.pi/2.)                                                                                                          
    C13=Rz(-np.pi/2.)                                                                                                                            
    C14=Ry(np.pi)*Rz(-np.pi/2.)                                                                                                            
    C15=Ry(-np.pi/2.)*Rz(-np.pi/2.)                                                                                                        
    C16=Ry(np.pi/2.)*Rz(-np.pi/2.)                                                                                                         
    C17=Rz(-np.pi/2.)*Rx(np.pi/2.)*Rz(np.pi/2.)
    C18=Rz(np.pi/2.)*Rx(np.pi/2.)*Rz(np.pi/2.)                                                                                       
    C19=Rz(np.pi)*Rx(np.pi/2.)*Rz(np.pi/2.)                                                                                          
    C20=Rx(np.pi/2.)*Rz(np.pi/2.)                                                                                                          
    C21=Rz(np.pi/2.)*Rx(-np.pi/2.)*Rz(np.pi/2.)                                                                                      
    C22=Rz(-np.pi/2.)*Rx(-np.pi/2.)*Rz(np.pi/2.)                                                                                     
    C23=Rx(-np.pi/2.)*Rz(np.pi/2.)                                                                                                         
    C24=Rx(np.pi)*Rx(-np.pi/2.)*Rz(np.pi/2.)

    C=[C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24]

    return C


def Clifford_group_generators(n):

    '''
    Outputs the generators of the n-qubit Clifford group.
    '''

    G=[]

    if n==1:
        G=[H_i(1,1),S_i(1,1)]
    else:
        for i in range(1,n+1):
            G.append(H_i(i,n))
            G.append(S_i(i,n))
            for j in range(1,n+1):
                if i<j:
                    G.append(CNOT_ij(i,j,n))
                else:
                    continue
    
    return G


def generate_Clifford_group(n,display=False):
    
    '''
    Generates the n-qubit Clifford group. The display variable is for testing
    purposes, and to see the progress through the code.

    Note that even for n=2, this code will take a long time to run! There are 
    11520 elements of the two-qubit Clifford group!
    '''

    G=Clifford_group_generators(n)
    
    def in_list(L,elem):
    
        # Last modified: 27 June 2019
        
        '''
        Checks if the given unitary elem is in the list L.
        '''
        
        x=0
        for l in L:
            if np.around(unitary_distance(l,elem),10)==0:  # Check of the distance is zero (up to 10 decimal places)
                x=1
                break
        
        return x


    C=[eye(2**n)]
    generated=False

    while not generated:

        tmp=[]
        num_added=0

        for c in C:
            for g in G:
                t1=c*g
                t2=c*g.H
                
                # t1 and t2 might be the same, in which case we add only one of the two to the list (if needed).
                # Also, t1 and t2 might already by in tmp (up to global phase), so we need to check for that as well.
                if np.around(unitary_distance(t1,t2),10)==0:
                    if not in_list(C,t1) and not in_list(tmp,t1):
                        tmp.append(t1)
                        num_added+=1
                else:  # if t1 and t2 are different, add both to the list (if needed).
                    if not in_list(C,t1) and not in_list(tmp,t1):
                        tmp.append(t1)
                        num_added+=1
                    if not in_list(C,t2) and not in_list(tmp,t2):
                        tmp.append(t2)
                        num_added+=1
        
        if num_added>0:
            for t in tmp:
                C.append(t)
        else:
            generated=True

        if display:
            print(len(C))
            
    return C


def generate_state_2design(C,n,display=True):

    '''
    Takes the n-qubit Clifford gates provided in C and returns a
    corresponding state 2-design. This uses the fact that the Clifford
    gates (for any n) form a unitary 2-design, and that any unitary 
    t-design can be used to construct a state t-design.
    '''

    def in_list(L,elem):

        # Last modified: 28 June 2019
    
        '''
        Checks if the given pure state elem is in the list L.
        '''
        
        x=0
        
        for l in L:
            if np.around(trace_distance_pure_states(l,elem),10)==0:
                x=1
                break
        
        return x

    S=[ket(2**n,0)]

    for c in C:
        s_test=c*ket(2**n,0)

        if not in_list(S,s_test):
            S.append(s_test)

        if display:
            print(len(S))

    return S



####################################################################
## su(D) LIE ALGEBRA
####################################################################


def su_generators(d):

    '''
    Generates the basis (aka generators) of the Lie algebra su(d)
    corresponding to the Lie group SU(d). The basis has d^2-1 elements.

    All of the generators are traceless and Hermitian. After adding the
    identity matrix, they form an orthogonal basis for all dxd matrices.

    The orthogonality condition is

        Tr[l_i*l_j]=d*delta_{i,j}

    (This is a particular convention we use here; there are other conventions.)

    For d=2, we get the Pauli matrices.
    '''

    L=[]

    L.append(eye(d))

    for l in range(d):
        for k in range(l):
            L.append(np.sqrt(d/2)*(ket(d,k)*ket(d,l).H+ket(d,l)*ket(d,k).H))
            L.append(np.sqrt(d/2)*(-1j*ket(d,k)*ket(d,l).H+1j*ket(d,l)*ket(d,k).H))

    for k in range(1,d):
        X=0
        for j in range(k):
            X+=ket(d,j)*ket(d,j).H
        
        L.append(np.sqrt(d/(k*(k+1)))*(X-k*ket(d,k)*ket(d,k).H))

    
    return L


def su_structure_constants(d):

    '''
    Generates the structure constants corresponding to the su(d)
    basis elements. They are defined as follows:

        f_{i,j,k}=(1/(1j*d^2))*Tr[l_k*[l_i,l_j]]

        g_{i,j,k}=(1/d^2)*Tr[l_k*{l_i,l_j}]
    
    '''

    f={}
    g={}

    L=su_generators(d)

    for i in range(1,d**2):
        for j in range(1,d**2):
            for k in range(1,d**2):

                f[(i,j,k)]=(1/(1j*d**2))*np.trace(L[k]*(L[i]*L[j]-L[j]*L[i]))

                g[(i,j,k)]=(1/d**2)*np.trace(L[k]*(L[i]*L[j]+L[j]*L[i]))

    return f,g


def su_generators_products(d):

    '''
    Generates a dictionary containing the span of the pairwise products of the 
    su(d) generators.

    P[(i,j)] gives a list of the indices of the su(d) generators such that

    L[i]*L[j]=\sum_k c_k L_k,

    where the sum is over the elements in P[(i,j)].
    '''

    L=su_generators(d)
    P={}

    for i in range(1,d**2):
        for j in range(1,d**2):
            P[(i,j)]=[]
            if i==j:
                P[(i,j)].append(0)
            for k in range(1,d**2):
                if np.trace(L[k]*L[i]*L[j])!=0:
                    P[(i,j)].append(k)
    
    return P


def state_from_coherence_vector(n,d,state=True):

    '''
    Uses the supplied coherence vector n to generate the corresponding state via

        rho=(1/d)*(eye(d)+n*L),

    where L are the su(d) generators.

    If state=True, then n is a vector of length d^2-1. Otherwise, if state=False, then n is a vector of length d^2
    '''

    L=su_generators(d)

    if state:
        rho=np.matrix((1/d)*eye(d),dtype=np.complex128)
        for i in range(1,len(L)):
            #print(rho)
            rho+=(1/d)*n[i-1]*L[i]
        return rho
    
    else:
        X=np.matrix(np.zeros((d,d)),dtype=np.complex128)
        for i in range(len(L)):
            X+=(1/d)*n[i]*L[i]
        return X


def coherence_vector_star_product(n1,n2,d):

    '''
    Computes the star product between two coherence vectors corresponding to states, so that
    n1 and n2 (the coherence vectors) have length d^2-1 each.

    Definition taken from:
    
        "Characterization of the positivity of the density matrix in terms of
        the coherence vector representation"
        PHYSICAL REVIEW A 68, 062322 (2003)
    '''

    #L=su_generators(d)
    g=su_structure_constants(d)[1]

    p=[]

    for k in range(1,d**2):
        pk=0
        for i in range(1,d**2):
            for j in range(1,d**2):
                pk+=(d/2)*n1[i-1]*n2[j-1]*g[(i,j,k)]
        p.append(pk)

    return np.array(p)

    


####################################################################
## DISCRETE WEYL OPERATORS
####################################################################


def discrete_Weyl_X(d):

    '''
    Generates the X shift operators.
    '''

    X=ket(d,1)*ket(d,0).H

    for i in range(1,d):
        X=X+ket(d,(i+1)%d)*ket(d,i).H

    return X

def discrete_Weyl_Z(d):

    '''
    Generates the Z phase operators.
    '''

    w=np.exp(2*np.pi*1j/d)

    Z=ket(d,0)*ket(d,0).H

    for i in range(1,d):
        Z=Z+w**i*ket(d,i)*ket(d,i).H

    return Z

def discrete_Weyl(d,a,b):

    '''
    Generates the discrete Weyl operator X^aZ^b.
    '''

    return discrete_Weyl_X(d)**a*discrete_Weyl_Z(d)**b


def nQudit_Weyl_coeff(X,d,n):

    '''
    Generates the coefficients of the operator X acting on n qudit
    systems.
    '''

    C={}

    S=list(itertools.product(*[range(0,d)]*n))

    for s in S:
        s=list(s)
        for t in S:
            t=list(t)
            G=generate_nQudit_X(d,s)*generate_nQudit_Z(d,t)
            C[(str(s),str(t))]=np.around(np.trace(X.H*G),10)

    return C


def generate_nQudit_X(d,indices):

    '''
    Generates a tensor product of discrete Weyl-X operators. indices is a 
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    '''

    X=discrete_Weyl_X(d)

    out=1

    for index in indices:
        out=tensor(out,X**index)

    return out


def generate_nQudit_Z(d,indices):

    '''
    Generates a tensor product of discrete Weyl-Z operators. indices is a
    list of dits (i.e., each element of the list is a number between 0 and
    d-1).
    '''

    Z=discrete_Weyl_Z(d)

    out=1

    for index in indices:
        out=tensor(out,Z**index)

    return out


def nQudit_quadratures(d,n):

    '''
    Returns the list of n-qudit "quadrature" operators, which are defined as
    (for two qudits)

        S[0]=X(0) \otimes Id
        S[1]=Z(0) \otimes Id
        S[2]=Id \otimes X(0)
        S[3]=Id \otimes Z(0)

    In general, for n qubits:

        S[0]=X(0) \otimes Id \otimes ... \otimes Id
        S[1]=Z(0) \otimes Id \otimes ... \otimes Id
        S[2]=Id \otimes X(0) \otimes ... \otimes Id
        S[3]=Id \otimes Z(0) \otimes ... \otimes Id
        .
        .
        .
        S[2n-2]=Id \otimes Id \otimes ... \otimes X(0)
        S[2n-1]=Id\otimes Id \otimes ... \otimes Z(0)
    '''

    S={}

    count=0

    for i in range(1,2*n+1,2):
        v=list(np.array(ket(n,count).H,dtype=np.int).flatten())
        S[i]=generate_nQudit_X(d,v)
        S[i+1]=generate_nQudit_Z(d,v)
        count+=1

    return S


def nQudit_cov_matrix(X,d,n):

    '''
    Generates the matrix of second moments (aka covariance matrix) of an
    n-qudit operator X.
    '''


    S=nQudit_quadratures(d,n)

    V=np.matrix(np.zeros((2*n,2*n)),dtype=np.complex128)

    for i in range(2*n):
        for j in range(2*n):
            #V[i,j]=np.trace(X*(S[i+1]*S[j+1].H+S[j+1].H*S[i+1]))
            V[i,j]=np.trace(X*S[i+1]*S[j+1].H)  # Use this instead to be consistent with the qubit case above.
    

    return V



####################################################################
## MISCELLANEOUS
####################################################################


def base_number_to_int(string,base):

    # Last modified: 10 August 2018

    b=base

    string=string[::-1]

    return sum([string[k]*b**k for k in range(len(string))])


def proj(u,v):

    '''
    Calculates the projection of vector v onto vector u.
    '''

    return (complex(u.H*v)/float(norm(u)**2))*u

def gram_schmidt(states,dim,normalize=True):

    '''
    Performs the Gram-Schmidt orthogonalization procedure on the given states
    (or simply vectors). dim is the dimension of the vectors.
    '''

    e=[]
    u=[]
    u.append(states[0])
    e.append(states[0]/norm(states[0]))

    for k in range(1,len(states)):
        S=np.matrix(np.zeros([dim,1]),dtype=complex)
        for j in range(k):
            S+=proj(u[j],states[k])
        u.append(states[k]-S)
        e.append(u[k]/norm(u[k]))
    
    if normalize==True:
        return e
    else:
        return u