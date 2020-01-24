'''
This code is part of QuTIPy.

(c) Copyright Sumeet Khatri, 2020


[license information?]
'''


from .base import *
from .channels import apply_channel





####################################################################
## STATE ENTROPIES
####################################################################



def bin_entropy(p):

    '''CURRENTLY NOT FUNCTIONING PROPERLY
    Returns the binary entropy for 0<=p<=1.
    '''

    if p==0:
        return 0
    elif p==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)


def entropy(rho):

    '''
    Returns the quantum (von Neumann) entropy of the state rho.
    '''

    return -np.real(np.trace(rho*np.matrix(logm(rho))))/np.log(2)


def relative_entropy(rho,sigma):

    '''
    Computes the standard (von Neumann) quantum relative entropy of rho
    and sigma, provided that supp(rho) is contained in supp(sigma).
    '''

    return np.real(np.trace(rho*(np.matrix(logm(rho))-np.matrix(logm(sigma)))))/np.log(2)


def relative_entropy_var(rho,sigma):

    '''
    Returns the relative entropy variance of rho and sigma, defined as

    V(rho||sigma)=Tr[rho*(log2(rho)-log2(sigma))^2]-D(rho||sigma)^2.
    '''

    return np.real(np.trace(rho*(np.matrix(logm(rho))/np.log(2)-np.matrix(logm(sigma))/np.log(2))**2))-relative_entropy(rho,sigma)**2


def mutual_information(rhoAB,dimA,dimB):

    '''
    Computes the mutual information of the bipartite state rhoAB, defined as

    I(A;B)_rho=D(rhoAB||rhoA\otimes rhoB)
    '''

    rhoA=TrX(rhoAB,[2],[dimA,dimB])
    rhoB=TrX(rhoAB,[1],[dimA,dimB])

    return relative_entropy(rhoAB,tensor(rhoA,rhoB))


def Holevo_inf_ensemble(p,S):

    '''
    Computes the Holevo information of an ensemble.

    p is an array of probabilities, and S is an array of states.

    Based on MATLAB code written by Felix Lediztky.
    '''

    dim=np.shape(S[0])[0]

    R=np.matrix(np.zeros((dim,dim)))
    av=0

    for i in range(len(p)):
        R=R+p[i]*S[i]
        av=av+p[i]*entropy(S[i])

    return entropy(R)-av


def coherent_inf_state(rho_AB,dimA,dimB,s=1):

    '''
    Calculates the coherent information of the state rho_AB.

    If s=2, then calculates the reverse coherent information.
    '''

    if s==1: # Calculate I_c(A>B)=H(B)-H(AB)
        rho_B=TrX(rho_AB,[1],[dimA,dimB])
        return entropy(rho_B)-entropy(rho_AB)
    else: # Calculate I_c(B>A)=H(A)- H(AB) (AKA reverse coherent information)
        rho_A=TrX(rho_AB,[2],[dimA,dimB])
        return entropy(rho_A)-entropy(rho_AB)



def Petz_Renyi_rel_ent(rho,sigma,alpha):

    '''
    Computes the Petz-Renyi relative entropy of rho and sigma for 0<=alpha<=1.
    '''

    rho_a=np.matrix(fractional_matrix_power(rho,alpha))
    sigma_a=np.matrix(fractional_matrix_power(sigma,1-alpha))

    Q=np.real(np.trace(rho_a*sigma_a))

    return (1./(alpha-1))*np.log2(Q)


def Petz_Renyi_mut_inf_state(rhoAB,dimA,dimB,alpha,opt=True):

    '''
    Computes the Petz-Renyi mutual information of the bipartite state
    rhoAB for 0<=alpha<=1.

    TO DO: Figure out how to do the computation with optimization over sigmaB.
    '''

    rhoA=TrX(rhoAB,[2],[dimA,dimB])
    rhoB=TrX(rhoAB,[1],[dimA,dimB])

    if opt==False:
        return Petz_Renyi_rel_ent(rhoAB,tensor(rhoA,rhoB),alpha)
    else:
        return None


def sandwiched_Renyi_rel_ent(rho,sigma,alpha):

    '''
    Computes the sandwiched Renyi relative entropy for either 0<=alpha<=1,
    or for alpha>=1 provided that supp(rho) is contained in supp(sigma).
    '''

    sigma_a=np.matrix(fractional_matrix_power(sigma,(1.-alpha)/(2*alpha)))

    Q=np.real(np.trace(fractional_matrix_power(sigma_a*rho*sigma_a,alpha)))

    return (1./(alpha-1))*np.log2(Q)


def sandwiched_Renyi_mut_inf_state(rhoAB,dimA,dimB,alpha,opt=True):

    '''
    Computes the sandwiched Renyi mutual information of the bipartite state
    rhoAB for 0<=alpha<=infty.

    TO DO: Figure out how to do the computation with optimization over sigmaB.
    '''

    rhoA=TrX(rhoAB,[2],[dimA,dimB])
    rhoB=TrX(rhoAB,[1],[dimA,dimB])

    if opt==False:
        return sandwiched_Renyi_rel_ent(rhoAB,tensor(rhoA,rhoB),alpha)
    else:
        return None


def hypo_testing_rel_ent(rho,sigma,eps,display=True):

    '''
    Calculates the eps-hypothesis testing relative entropy of the two states
    rho and sigma.

    CURRENTLY NOT FUNCTIONING PROPERLY

    '''

    dim=rho.shape[0]

    L=cvx.Variable((dim,dim),hermitian=True)

    c=[]
    c+=[L>>0,eye(dim)-L>>0]
    
    #c+=[cvx.real(cvx.trace(cvx.matmul(L,rho)))>=1-eps]
    c+=[cvx.trace(cvx.matmul(L,rho))>=1-eps]

    obj=cvx.Minimize(cvx.trace(cvx.matmul(L,sigma)))
    prob=cvx.Problem(obj,c)
    prob.solve(solver=cvx.CVXOPT,eps=1e-9,verbose=display)

    return -np.log2(prob.value)


def hypo_testing_rel_ent_dual(rho,sigma,eps,display=True):

    '''
    Computes the dual of the SDP for the hypothesis testing relative entropy.

    CURRENTLY NOT FUNCTIONING PROPERLY
    '''

    dim=rho.shape[0]

    Y=cvx.Variable((dim,dim),hermitian=True)
    l=cvx.Variable()

    c=[]
    #c+=[Y>>0,l>=0,Y+sigma-l*rho>>0]
    #obj=cvx.Maximize(cvx.real(-cvx.trace(Y)+l*(1-eps)))

    c+=[l>=0,Y>>sigma,Y-l*rho>>0]
    obj=cvx.Maximize(-cvx.trace(Y)+cvx.trace(sigma)+l*(1-eps))

    prob=cvx.Problem(obj,c)
    prob.solve(solver=cvx.CVXOPT,eps=1e-9,verbose=display)

    return -np.log2(prob.value)









####################################################################
## CHANNEL ENTROPIES
####################################################################


def Holevo_inf_channel(K,dim,display=True):

    '''
    Computes the Holevo information of a channel given by its set of
    Kraus operators K. dim is the dimension of the input space of the
    channel.

    Based on MATLAB code written by Felix Leditzky.
    '''


    def objfunc(x):

        Re=np.matrix(x[0:dim**3])
        Im=np.matrix(x[dim**3:])

        psi=np.matrix(Re.T+1j*Im.T)
        psi=psi/norm(psi)

        p=[]
        S=[]

        for j in range(dim**2):
            R=tensor(ket(dim**2,j),eye(dim)).H*(psi*psi.H)*tensor(ket(dim**2,j),eye(dim))
            p.append(np.trace(R))
            rho=R/np.trace(R)
            rho_out=apply_channel(K,rho)
            S.append(rho_out)
        
        return -np.real(Holevo_inf_ensemble(p,S))

    
    x_init=np.random.rand(2*dim**3)

    opt=minimize(objfunc,x_init,options={'disp':display})

    return -opt.fun


def coherent_inf_channel(K,dim_in,dim_out,s=1,display=True):

    '''
    Calculates the coherent information of the channel specified by
    the Kraus operators in K.

    If s=2, then calculates the reverse coherent information of the channel.
    '''


    def objfunc(x):

        Re=np.matrix(x[0:dim_in**2])
        Im=np.matrix(x[dim_in**2:])

        psi=np.matrix(Re.T+1j*Im.T)
        psi=psi/norm(psi)

        psi_AA=psi*psi.H

        rho_AB=apply_channel(K,psi_AA,2,dim=[dim_in,dim_in])

        return -coherent_inf_state(rho_AB,dim_in,dim_out,s)


    x_init=np.random.rand(2*dim_in**2)

    opt=minimize(objfunc,x_init,options={'disp':display})

    return np.max([0,-opt.fun])





