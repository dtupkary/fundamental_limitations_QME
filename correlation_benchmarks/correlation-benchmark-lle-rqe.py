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
import os.path
import sys
import time


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
    N=3
    w0max=1
    w0min=1
    gmin=g
    gmax=g
    
    w0list=np.linspace(w0min,w0max,N)
    glist=np.linspace(gmin,gmax,N-1)
    
    tb=0.01
    epsilon=0.1
    gamma1=1 #gamma1 is the coupling to left bath. It shows up in spectral bath function
    gamma2=1    #gamma2 is the coupling to the right bath.    
    
    
    
    beta=1
    mu=-0.5
    
    
    delta=0
    mu1=mu
    mu2=mu
    beta1=beta
    beta2=beta
    
       
    
    H_S=create_hamiltonian(w0list,glist,delta,N)
    
    
    c_N=create_sm(N,N)  # we couple the Nth spin to the bath
    c_1=create_sm(N,1)
    
    
    
    eigenergies,eigstates=H_S.eigenstates()
    
    #print("eigenenergies are : ",eigenergies)
    
    spectrum=max(eigenergies)-min(eigenergies)
    
    print("max energy , min energy = ", max(eigenergies),min(eigenergies))
    
    number=len(eigenergies)
    
    integral11=np.empty((number,number),dtype=np.cdouble) #stores J * N integral for left bath
    integral12=np.empty((number,number),dtype=np.cdouble) # stores J integral (just to check) for the left bath
    integral21=np.empty((number,number),dtype=np.cdouble) #stores J*N integral for right bath
    integral22=np.empty((number,number),dtype=np.cdouble)
    
    
    
    
    for i in range(number):
        for k in range(number):
            freq=eigenergies[k]-eigenergies[i]
            #print(i,k,freq)
            if( np.absolute(freq) >= 1/10**10):
                integral11[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta1,mu1,gamma1),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
                integral12[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma1),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0]  #left bath done
                integral21[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta2,mu2,gamma2),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
                integral22[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma2),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0]  #right bath
    
            if (np.absolute(freq)<=1/10**10):
                integral11[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func2,0,b,args=(tb,beta1,mu1,gamma1),limit=limit_value)[0]
                integral12[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath_2,0,b,args=(tb,gamma1),limit=limit_value)[0]
                integral21[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func2,0,b,args=(tb,beta2,mu2,gamma2),limit=limit_value)[0]
                integral22[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath_2,0,b,args=(tb,gamma2),limit=limit_value)[0]
             
            
            #expected=1.0j*(eigenergies[k]-eigenergies[i])/(2*tb*tb)
    #        print(i,k,integral2[i,k],expected)
    
    
    # PAY ATTENTION TO THE WAY THESE COEFFICIENTS ARE BEING COMPUTED
    
    constant12=np.empty((number,number),dtype=np.cdouble)
    constant11=np.empty((number,number),dtype=np.cdouble)
    constant21=np.empty((number,number),dtype=np.cdouble)
    constant22=np.empty((number,number),dtype=np.cdouble)
    
    
    
    for i in range(number):
        for k in range(number):
            constant12[i,k]=integral12[i,k]+integral11[i,k]+0.5*(spectral_bath(eigenergies[k]-eigenergies[i],tb,gamma1)+func1(eigenergies[k]-eigenergies[i],tb,beta1,mu1,gamma1))    #full coefficient created this is nbar+1
            constant11[i,k]=integral11[i,k]+0.5*func1(eigenergies[k]-eigenergies[i],tb,beta1,mu1,gamma1)                                       # the full coefficient is created
            
            constant22[i,k]=integral22[i,k]+integral21[i,k]+0.5*(spectral_bath(eigenergies[k]-eigenergies[i],tb,gamma2)+func1(eigenergies[k]-eigenergies[i],tb,beta2,mu2,gamma2))    #full coefficient created this is nbar+1
            constant21[i,k]=integral21[i,k]+0.5*func1(eigenergies[k]-eigenergies[i],tb,beta2,mu2,gamma2)   # the full coefficient is created
            #print(i,k,constant11[i,k],constant12[i,k],constant21[i,k],constant22[i,k])
    list1=[]
    list2=[]
    
    
    for i in range(number):
        list1.append([])
        list2.append([])
    
    
    
    matrix=np.zeros((number,number))
    
    dim=[]
    for k in range(N):
        dim.append(2)    
    
    zeromatrix=Qobj(matrix,dims=[dim,dim])
    
    
    indices1=[]
    indices2=[]
    
    
    
    
    for i in range(number):
        for k in range(number):
            list1[i].append(eigstates[i]*eigstates[i].dag()*c_1*eigstates[k]*eigstates[k].dag())
            list2[i].append(eigstates[i]*eigstates[i].dag()*c_N*eigstates[k]*eigstates[k].dag())
            
            if(tracedist(eigstates[i]*eigstates[i].dag()*c_1*eigstates[k]*eigstates[k].dag(),zeromatrix)!=0):
                indices1.append((i,k))
            if(tracedist(eigstates[i]*eigstates[i].dag()*c_N*eigstates[k]*eigstates[k].dag(),zeromatrix)!=0):
                indices2.append((i,k))
    
    
    
    
    
    #### NOW WE START THE LIOVILLE SHIT. 
    
    
    pre=-1.0j*H_S
    post=1.0j*H_S
    
    L=spre(pre)+spost(post)
    
    for i in range(number):
        for k in range(number):
            vi=eigstates[i]
            vk=eigstates[k]
            
            op1=epsilon*epsilon*constant11[i,k]*vi*vi.dag()*c_1*vk*vk.dag()*c_1.dag()
            op2=epsilon*epsilon*constant12[i,k]*c_1.dag()*vi*vi.dag()*c_1*vk*vk.dag()
            
            op3=epsilon*epsilon*constant11[i,k]*c_1.dag()
            op4=vi*vi.dag()*c_1*vk*vk.dag()
            op5=epsilon*epsilon*constant12[i,k]*c_1.dag()
            
            
            L=L+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L=L+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
            
            op1=epsilon*epsilon*constant21[i,k]*vi*vi.dag()*c_N*vk*vk.dag()*c_N.dag()
            op2=epsilon*epsilon*constant22[i,k]*c_N.dag()*vi*vi.dag()*c_N*vk*vk.dag()
            
            op3=epsilon*epsilon*constant21[i,k]*c_N.dag()
            op4=vi*vi.dag()*c_N*vk*vk.dag()
            op5=epsilon*epsilon*constant22[i,k]*c_N.dag()
            
            
            L=L+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L=L+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
            
            
            
    #Variables needed for for iterative-lgmres to work. 
    return_info=True
    
    ss_redfield,dict_redfield=steadystate(L,return_info=return_info)
       
    
   
    
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
    
    
    ss_lindblad,dict_lindblad=steadystate(H,Cops,return_info=return_info)
    
       
    
   
    
    
    ######### NEXT WE DO THE THEORY SHIT (if needed)
    mag=0.5*create_magnetization(N)
    theory=(-beta*(H_S-mu*mag)).expm()
    ss_theory=theory/theory.tr()
    
    L_redfield=L
    L_lindblad=liouvillian(H_S,Cops)
    
    
    
    
    
    
    ### WE Will check degeneracy now
   
    if (np.absolute(L_redfield.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in redfield")

    if (np.absolute(L_lindblad.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in local lindblad")

    #### We have computed the Redfield and locallindblad steady states, now we must move to the correlation matrix calculations.
    #### what follows will be based on the theory of the Lindblad equations. 
    #we will need the H_S in the site basis
    
    Hmatrix=np.zeros((N,N),dtype=np.complex64)
    A=np.zeros((N,N),dtype=np.complex64)
    S=np.zeros((N,N),dtype=np.complex64)
    
    M=np.zeros((N,N),dtype=np.complex64)
    m=np.zeros((N,N),dtype=np.complex64)
    
    Q=np.zeros((N,N),dtype=np.complex64)
    P=np.zeros((N,N),dtype=np.complex64)
    
    
    ##create Hmatrix first.
    
    
    
    integral_11=np.zeros((N),dtype=np.complex64)
    integral_12=np.zeros((N),dtype=np.complex64) 
    integral_22=np.zeros((N),dtype=np.complex64) 
    integral_22=np.zeros((N),dtype=np.complex64) 
    
    
    for k in range(N):
        
    
    
    