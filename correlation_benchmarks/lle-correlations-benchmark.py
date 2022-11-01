# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:56:50 2020
benchmark between correlations and full, for local-lindblad and REdfield appraches
@author: Devashish
"""




from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import scipy


def create_vacuum(N):
    psi=basis(2,1)
    for k in range(2,N+1):
        psi=tensor(psi,basis(2,1))

    return psi

def create_allup(N):
    psi=basis(2,0)
    for k in range(2,N+1):
        psi=tensor(psi,basis(2,0))    

    return psi



def create_magnetization(N):
    op=0

    for k in range(1,N+1):
        op=op+create_sigmaz(N,k)

    return op



def create_sm(N,pos): #creates the sigma_minus operator for the given position. N>=2
    if pos==1:
        op=create(2)
        for k in range(2,N+1):
            op=tensor(op,qeye(2))

    else:
        op=qeye(2)
        for k in range(2,N+1):
            if k==pos:
                op=tensor(op,create(2))
            else:
                op=tensor(op,qeye(2))

    return op


def create_sigmax(N,pos):
    if pos==1:
        op=sigmax()
        for k in range(2,N+1):
            op=tensor(op,qeye(2))

    else:
        op=qeye(2)
        for k in range(2,N+1):
            if k==pos:
                op=tensor(op,sigmax())
            else:
                op=tensor(op,qeye(2))
    return op


def create_sigmay(N,pos):
    if pos==1:
        op=sigmay()
        for k in range(2,N+1):
            op=tensor(op,qeye(2))

    else:
        op=qeye(2)
        for k in range(2,N+1):
            if k==pos:
                op=tensor(op,sigmay())
            else:
                op=tensor(op,qeye(2))
    return op


def create_sigmaz(N,pos):
    if pos==1:
        op=sigmaz()
        for k in range(2,N+1):
            op=tensor(op,qeye(2))

    else:
        op=qeye(2)
        for k in range(2,N+1):
            if k==pos:
                op=tensor(op,sigmaz())
            else:
                op=tensor(op,qeye(2))

    return op



def create_hamiltonian(w0list,glist,delta,N):
    
    H=(w0list[N-1]/2)*create_sigmaz(N,N)

    for k in range(1,N):
        H=H+(w0list[k-1]/2)*create_sigmaz(N,k) - glist[k-1]*(create_sigmax(N,k)*create_sigmax(N,k+1) + create_sigmay(N,k)*create_sigmay(N,k+1) + delta*create_sigmaz(N,k)*create_sigmaz(N,k+1))

    return H



def spectral_bath(omega,tb,gamma):
    if(omega <=0):
        return 0

    return gamma*omega*np.exp(-omega*omega*tb)



def spectral_bath_2(omega,tb,gamma):
    if(omega <=0):
        return 0
    return gamma*np.exp(-omega*omega*tb)
    



def nbar(omega,beta,mu):
    return 1/(np.exp(beta*(omega-mu))-1)


def func1(omega,tb,beta,mu,gamma):
    if(omega <=0):
        return 0

    return spectral_bath(omega,tb,gamma)*nbar(omega,beta,mu)



def func2(omega,tb,beta,mu,gamma):
    if(omega<=0):
        return 0

    return spectral_bath_2(omega,tb,gamma)*nbar(omega,beta,mu)



def evolve(state,H_S,list1,list2,C11,C12,C21,C22,number,epsilon,c_1,c_N,indices1,indices2):
    term1=1.0j*commutator(state,H_S)
    
    term2=0

    for index in indices1:
        i=index[0]
        k=index[1]
        term2=term2+commutator(state*list1[i][k],c_1.dag())*C11[i,k]
        term2=term2+commutator(c_1.dag(),list1[i][k]*state)*C12[i,k]
        
    for index in indices2:
        i=index[0]
        k=index[1]
        term2=term2+commutator(state*list2[i][k],c_N.dag())*C21[i,k]
        term2=term2+commutator(c_N.dag(),list2[i][k]*state)*C22[i,k]


    return term1-epsilon*epsilon*(term2+term2.dag())



def convergencecheck(array):
    length=len(array)
    maxv=np.max(array)
    minv=np.min(array)
    mean=np.mean(array)
    

    #print(max,min,"mean =",mean)
    if ((maxv-minv) < 0.05*abs(mean) and (maxv-minv) > -0.05*abs(mean)):
        return 1
    
    return 0




#declaring parameters
#
gvals=[0.2]


for g in gvals:
        
    
    print("g value is ",g )
    b=50
    limit_value=700
    N=4
    w0max=1
    w0min=1
    w0=1
    gmin=g
    gmax=g
    
    w0list=np.linspace(w0min,w0max,N)
    glist=np.linspace(gmin,gmax,N-1)
    
    tb=0.01
    epsilon=0.1
    gamma1=0 #gamma1 is the coupling to left bath. It shows up in spectral bath function
    gamma2=1   #gamma2 is the coupling to the right bath.    
    
    
    
    beta=1
    mu=-0.5
    
    
    delta=0
    mu1=-0.1
    mu2=-0.8
    beta1=5
    beta2=1
    
    
       
    
    H_S=create_hamiltonian(w0list,glist,delta,N)
    
    
    c_N=create_sm(N,N)  # we couple the Nth spin to the bath
    c_1=create_sm(N,1)
    
    
    
    eigenergies,eigstates=H_S.eigenstates()
    
    #print("eigenenergies are : ",eigenergies)
    
    spectrum=max(eigenergies)-min(eigenergies)
    
    
   
    
    ################## Local lindbald shit #############################
    
    
    Delta1=(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma1),weight='cauchy',wvar=w0list[0])[0] #Delta
    Deltadash1=(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta1,mu1,gamma1),weight='cauchy',wvar=w0list[0])[0] #Delta
    
    
    DeltaN=(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma2),weight='cauchy',wvar=w0list[N-1])[0] #Delta
    DeltadashN=(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta2,mu2,gamma2),weight='cauchy',wvar=w0list[N-1])[0] #Delta
    
    
    H=H_S+(Deltadash1+0.5*Delta1)*create_sigmaz(N,1)+(DeltadashN+0.5*DeltaN)*create_sigmaz(N,N)
    
    
    Cops=[]
    
    Cops.append(epsilon*np.sqrt(func1(w0list[0],tb,beta1,mu1,gamma1)+spectral_bath(w0list[0], tb, gamma1))*create_sm(N,1))
    Cops.append(epsilon*np.sqrt(func1(w0list[0],tb,beta1,mu1,gamma1))*create_sm(N, 1).dag())
    Cops.append(epsilon*np.sqrt(func1(w0list[N-1],tb,beta2,mu2,gamma2)+spectral_bath(w0list[N-1],tb,gamma2))*create_sm(N,N));
    Cops.append(epsilon*np.sqrt(func1(w0list[N-1],tb,beta2,mu2,gamma2))*create_sm(N,N).dag())
    
    L_lindblad=liouvillian(H,Cops)
    
    return_info=True
    ss_lindblad,dict_lindblad=steadystate(H,Cops,return_info=return_info)
    
       
    
   
    
    
 
    
    
    
    
     #### We have computed the locallindblad steady states, now we must move to the correlation matrix calculations.
    #### what follows will be based on the theory of the Lindblad equations. 
    #we will need the H_S in the site basis
    
    Hmatrix=np.zeros((N,N),dtype=np.complex64)
    A=np.zeros((N,N),dtype=np.complex64)
    S=np.zeros((N,N),dtype=np.complex64)
      
    
    ##create Hmatrix first.
    
    
    Hm=np.zeros((N,N),dtype=np.complex128)
    for i in range(N-1):
        Hm[i,i]=w0list[i]
        Hm[i,i+1]=-2*g
        Hm[i+1,i]=-2*g
    Hm[N-1,N-1]=w0list[N-1]    
    
   # Hm[0,0]=H[0,0]+(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma1),weight='cauchy',wvar=w0list[0])[0]
   # Hm[N-1,N-1]=H[N-1,N-1]+(-1.0*epsilon*epsilon/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma2),weight='cauchy',wvar=w0list[N-1])[0]
    Hm[0,0]=Hm[0,0]+Delta1+2*Deltadash1
    Hm[N-1,N-1]=Hm[N-1,N-1]+DeltaN+2*DeltadashN
    
    
    A[0,0]=-0.5*epsilon*epsilon*(2*func1(w0list[0],tb,beta1,mu1,gamma1)+spectral_bath(w0list[0],tb,gamma1))
    A[N-1,N-1]=-0.5*epsilon*epsilon*(2*func1(w0list[N-1],tb,beta2,mu2,gamma2)+spectral_bath(w0list[N-1],tb,gamma2))
    
    
    S[0,0]=epsilon*epsilon*func1(w0list[0],tb,beta1,mu1,gamma1)
    S[N-1,N-1]=epsilon*epsilon*func1(w0list[N-1],tb,beta2,mu2,gamma2)
    
    
    ss_sitebasis=scipy.linalg.solve_lyapunov(1.0j*Hm+A,-S)
    
    
    vec_full=np.zeros((2*N-1,1),dtype=np.complex64)
    vec_corr=np.zeros((2*N-1,1),dtype=np.complex64)
    vec_theory=np.zeros((2*N-1,1),dtype=np.complex64)

    
    vec_index=0
    for j in range(1,N+1):
        vec_full[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j),ss_lindblad)
        vec_corr[vec_index]=ss_sitebasis[j-1,j-1]
        vec_index=vec_index+1
    for j in range(1,N):
        vec_full[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j+1),ss_lindblad)
        vec_corr[vec_index]=ss_sitebasis[j-1,j]
        vec_index=vec_index+1
    
    
    print('max difference between full and corr  ',np.max(np.absolute(vec_full-vec_corr)))
   
    