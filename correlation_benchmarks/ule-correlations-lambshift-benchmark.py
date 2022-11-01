#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:46:37 2020
Code to compute the difference between full ULE and correlations approach INCLUDING LAMB SHIFt and excluding it.
@author: devashish
"""

from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import time
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



def convergencecheck(array):
    length=len(array)
    max=np.max(array)
    min=np.min(array)
    mean=np.mean(array)
    

    #print(max,min,"mean =",mean)
    if ((max-min) < 0.05*abs(mean) and (max-min) > -0.05*abs(mean)):
        return 0
    
    return 1


def gmatrix(omega, gamma, beta, mu, tb, i , j): #computes the i,j the element of the submatrix of the lth bath. 
    # l is determined by beta, mu and tb etc
    
    submatrix=np.zeros((2,2),dtype=np.complex128)
    submatrix[0,0]=1
    submatrix[0,1]=1.0j
    submatrix[1,0]=-1.0j
    submatrix[1,1]=1                                  
    if (omega <= 0):
        factor=np.sqrt(func1(-omega,tb,beta,mu,gamma)*2)/(8*np.pi)    
        return factor*submatrix[i-1,j-1]
    if (omega > 0):
        factor=np.sqrt(2*(func1(omega,tb,beta,mu,gamma)+spectral_bath(omega,tb,gamma)))/(8*np.pi)
        return factor*submatrix.conj()[i-1,j-1]

        
def lamb_integrand(omega,E1,E2,alpha,beta,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,flag): #this function computes the integrand of the lamb shift formula.
    term=0
    
    if (alpha <0 or beta<0):
        print('indices are wrong. ERror')
        return 0
    elif (alpha>4 or beta>4):
        print('indices are wrong. err0r')
        return 0
    #alpha, beta have to be between 1 and 4.
    elif (alpha <=2 and beta>=3): #we output zero
        return 0
    elif (alpha>=3 and beta <=2):
        return 0
    elif (alpha<=2 and beta<=2): #we are in the first bath setup
        term=term+gmatrix(omega-E1,gamma1,beta1,mu1,tb,alpha,1)*gmatrix(omega+E2,gamma1,beta1,mu1,tb,1,beta)
        term=term+gmatrix(omega-E1,gamma1,beta1,mu1,tb,alpha,2)*gmatrix(omega+E2,gamma1,beta1,mu1,tb,2,beta)
        if (flag==0):
            return term.real
        elif (flag==1):
            return term.imag
        else:
            print('flag invalid')
            return term
    elif (alpha >=3 and beta>=3):
        term=term+gmatrix(omega-E1,gamma2,beta2,mu2,tb,alpha-2,1)*gmatrix(omega+E2,gamma2,beta2,mu2,tb,1,beta-2)
        term=term+gmatrix(omega-E1,gamma2,beta2,mu2,tb,alpha-2,2)*gmatrix(omega+E2,gamma2,beta2,mu2,tb,2,beta-2)
        if (flag==0):
            return term.real
        elif (flag==1):
            return term.imag
        else:
            print('flag invalid')
            return term
#declaring parameters
#
#gvals=np.linspace(0.01,0.25,13)
gvals=[0.2]
#if this happens they open the output file at the same tiem and thinsg go wrong.
for g in gvals:
        
    print("g value is ",g )
    b=50
    
    N=3
    w0=1
    w0max=w0
    w0min=w0
    gmin=g
    gmax=g
    
    w0list=np.linspace(w0min,w0max,N)
    glist=np.linspace(gmin,gmax,N-1)
    
    tb=1
    epsilon=0.2
    gamma1=1 #gamma1 is the coupling to left bath. It shows up in spectral bath function
    gamma2=1    #gamma2 is the coupling to the right bath.    
    
    
    
    beta=0.2
    mu=-0.5
    
    
    delta=0
    mu1=-1.5
    mu2=mu
    beta1=2
    beta2=beta
    
       
    
    H_S=create_hamiltonian(w0list,glist,delta,N)
    
    
    c_N=create_sm(N,N)  # we couple the Nth spin to the bath
    c_1=create_sm(N,1)
    
    
    ### First we do the normal ULE Shit.
    eigenergies,eigstates=H_S.eigenstates()
    number=len(eigenergies)
    ######################## NOw we do universal lindblad shit. #######################################
    
    
    Cops_universal=[]
    
    #lambda= first bath, first op
    op=0
    for m in range(number):
        for n in range(number):
            Em=eigenergies[m]
            En=eigenergies[n]
            vm=eigstates[m]
            vn=eigstates[n]
            op=op+gmatrix(En-Em, gamma1, beta1, mu1, tb, 1 , 1)*vm*vm.dag()*create_sigmax(N,1)*vn*vn.dag()
            op=op+gmatrix(En-Em, gamma1, beta1, mu1, tb, 1, 2)*vm*vm.dag()*create_sigmay(N,1)*vn*vn.dag()
    
    Cops_universal.append(2*np.pi*epsilon*op)
    
    
    #lambda=first bath, second op 
    op=0
    for m in range(number):
        for n in range(number):
            Em=eigenergies[m]
            En=eigenergies[n]
            vm=eigstates[m]
            vn=eigstates[n]
            op=op+gmatrix(En-Em, gamma1, beta1, mu1, tb, 2 , 1)*vm*vm.dag()*create_sigmax(N,1)*vn*vn.dag()
            op=op+gmatrix(En-Em, gamma1, beta1, mu1, tb, 2, 2)*vm*vm.dag()*create_sigmay(N,1)*vn*vn.dag()
    
    Cops_universal.append(2*np.pi*epsilon*op)
    
    
    
    #lambda=secondbath bath, first op 
    op=0
    for m in range(number):
        for n in range(number):
            Em=eigenergies[m]
            En=eigenergies[n]
            vm=eigstates[m]
            vn=eigstates[n]
            op=op+gmatrix(En-Em, gamma2, beta2, mu2, tb, 1 , 1)*vm*vm.dag()*create_sigmax(N,N)*vn*vn.dag()
            op=op+gmatrix(En-Em, gamma2, beta2, mu2, tb, 1, 2)*vm*vm.dag()*create_sigmay(N,N)*vn*vn.dag()
    
    Cops_universal.append(2*np.pi*epsilon*op)
    
    
    #lambda=secondbath second op
    op=0
    for m in range(number):
        for n in range(number):
            Em=eigenergies[m]
            En=eigenergies[n]
            vm=eigstates[m]
            vn=eigstates[n]
            op=op+gmatrix(En-Em, gamma2, beta2, mu2, tb, 2 , 1)*vm*vm.dag()*create_sigmax(N,N)*vn*vn.dag()
            op=op+gmatrix(En-Em, gamma2, beta2, mu2, tb, 2, 2)*vm*vm.dag()*create_sigmay(N,N)*vn*vn.dag()
    
    Cops_universal.append(2*np.pi*epsilon*op)
    
    
    
    

# we try the lamb shift computation now. for that we need the function f 
# lamb_integrand(omega,E1,E2,alpha,beta,gamma1,gamma2,beta1,beta2,mu1,mu2,tb): #this function computes the integrand of the lamb shift formula.
               
    X_list=[]
    X_list.append(create_sigmax(N,1))
    X_list.append(create_sigmay(N,1))
    X_list.append(create_sigmax(N,N))
    X_list.append(create_sigmay(N,N))
    
    print('starting lambshift calculations.')
    lambshift=0
    for alpha_index in range(1,5):
        for beta_index in range(1,5):
            for l in range(number):
                for m in range(number):
                    for n in range(number):
                        Em=eigenergies[m]
                        En=eigenergies[n]
                        El=eigenergies[l]
                        factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(El-Em,En-El,alpha_index,beta_index,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
                        factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(El-Em,En-El,alpha_index,beta_index,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
                        term=(factor_real+factor_imag)*eigstates[m]*eigstates[m].dag()*X_list[alpha_index-1]*eigstates[l]*eigstates[l].dag()*X_list[beta_index-1]*eigstates[n]*eigstates[n].dag()
                        
                        lambshift=lambshift+term
        
    print('Lambshift for full ULE completed')
    ss_universal_withls=steadystate(H_S+lambshift,Cops_universal)
    ss_universal_withoutls=steadystate(H_S,Cops_universal)
    
################################### Now we do the correlations shit ####################################3
    H=np.zeros((N,N),dtype=np.complex128)
    for i in range(N-1):
        H[i,i]=w0
        H[i,i+1]=-2*g
        H[i+1,i]=-2*g
    H[N-1,N-1]=w0 
    
      
    #,next we diagonize the matrix and create phi.
    
    modes, Phi=np.linalg.eigh(H)  #phi is a matrix with ith column being ith eigenvector of H_{jk}

    H_eigenbasis=Phi.conjugate().transpose()@H@Phi
    
    # we now construct the two matrices m and M?
    
    M=np.zeros((N,N),dtype=np.complex64)
    m=np.zeros((N,N),dtype=np.complex64)
    
    # l =1
    for alpha in range(N):
        freq=modes[alpha]
        if (freq<=0):
            break
        else:
            M[0,alpha]=(epsilon*Phi[0,alpha]/np.sqrt(2))*np.sqrt(func1(freq,tb,beta1,mu1,gamma1)+spectral_bath(freq,tb,gamma1))
            m[0,alpha]=(epsilon*Phi[0,alpha].conjugate()/np.sqrt(2))*np.sqrt(func1(freq,tb,beta1,mu1,gamma1))
            
            M[N-1,alpha]=(epsilon*Phi[N-1,alpha]/np.sqrt(2))*np.sqrt(func1(freq,tb,beta2,mu2,gamma2)+spectral_bath(freq,tb,gamma2))
            m[N-1,alpha]=(epsilon*Phi[N-1,alpha].conjugate()/np.sqrt(2))*np.sqrt(func1(freq,tb,beta2,mu2,gamma2))
   
    V=m.conjugate().transpose()@m
    W=M.conjugate().transpose()@M
       
    Gamma1=np.zeros((N,N),dtype=np.complex128) #stores the lamb shift term.
    Gamma2=np.zeros((N,N),dtype=np.complex128)
    
    print('starting Gamma1 calculations')
    for alpha in range(N):
        for alphaprime in range(N):
                     
            term=0
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],1,1,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],1,1,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2*(factor_real+factor_imag)*Phi[0,alpha]*Phi[0,alphaprime].conjugate()
    
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],3,3,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],3,3,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2*(factor_real+factor_imag)*Phi[N-1,alpha]*Phi[N-1,alphaprime].conjugate()
            
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],1,2,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],1,2,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term-2.0j*(factor_real+factor_imag)*Phi[0,alpha]*Phi[0,alphaprime].conjugate()
            
            
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],3,4,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(modes[alpha],-modes[alphaprime],3,4,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term-2.0j*(factor_real+factor_imag)*Phi[N-1,alpha]*Phi[N-1,alphaprime].conjugate()
            
            Gamma1[alpha,alphaprime]=term
            
            term=0
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],1,1,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],1,1,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2*(factor_real+factor_imag)*Phi[0,alpha].conjugate()*Phi[0,alphaprime]
    
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],3,3,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],3,3,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2*(factor_real+factor_imag)*Phi[N-1,alpha].conjugate()*Phi[N-1,alphaprime]
            
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],modes[alphaprime],1,2,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],modes[alphaprime],1,2,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2.0j*(factor_real+factor_imag)*Phi[0,alpha].conjugate()*Phi[0,alphaprime]
            
            
            factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],3,4,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
            factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(-modes[alpha],+modes[alphaprime],3,4,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
            term=term+2.0j*(factor_real+factor_imag)*Phi[N-1,alpha].conjugate()*Phi[N-1,alphaprime]
            
            Gamma2[alpha,alphaprime]=term
            
            
          
            
            
            
    
    ss_eigenbasis_withls=scipy.linalg.solve_continuous_lyapunov(1.0j*H_eigenbasis-V-W.transpose()-1.0j*Gamma1+1.0j*Gamma2.transpose(),-2*V)
    ss_sitebasis_withls=Phi.conjugate()@ss_eigenbasis_withls@Phi.transpose()
    
    ss_eigenbasis_withoutls=scipy.linalg.solve_continuous_lyapunov(1.0j*H_eigenbasis-V-W.transpose(),-2*V)
    ss_sitebasis_withoutls=Phi.conjugate()@ss_eigenbasis_withoutls@Phi.transpose()
    
    
    vec_full_withls=np.zeros((2*N-1,1),dtype=np.complex64)
    vec_full_withoutls=np.zeros((2*N-1,1),dtype=np.complex64)
    vec_corr_withls=np.zeros((2*N-1,1),dtype=np.complex64)
    vec_corr_withoutls=np.zeros((2*N-1,1),dtype=np.complex64)
    
    vec_index=0
    for j in range(1,N+1):
        vec_full_withls[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j),ss_universal_withls)
        vec_full_withoutls[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j),ss_universal_withoutls)
        vec_corr_withls[vec_index]=ss_sitebasis_withls[j-1,j-1]
        vec_corr_withoutls[vec_index]=ss_sitebasis_withoutls[j-1,j-1]
        vec_index=vec_index+1
    for j in range(1,N):
        vec_full_withls[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j+1),ss_universal_withls)
        vec_full_withoutls[vec_index]=expect(create_sm(N,j).dag()*create_sm(N,j+1),ss_universal_withoutls)
        vec_corr_withls[vec_index]=ss_sitebasis_withls[j-1,j]
        vec_corr_withoutls[vec_index]=ss_sitebasis_withoutls[j-1,j]
        vec_index=vec_index+1
                       
                       
        
    print('max difference between full and corr , without LS ',np.max(np.absolute(vec_full_withoutls-vec_corr_withoutls)))
        
    print('max difference between full and corr , with LS ',np.max(np.absolute(vec_full_withls-vec_corr_withls)))