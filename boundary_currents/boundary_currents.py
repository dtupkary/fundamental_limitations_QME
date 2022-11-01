# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:56:50 2020
Computes Current vs g. Also computes boundary current at the end, 
@author: Devashish
"""
import os.path
import sys
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
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
    max_v=np.max(array)
    min_v=np.min(array)
    mean=np.mean(array)
    

    #print(max,min,"mean =",mean)
    if (max_v-min_v < 1e-9):
        return 0
    if ((max_v-min_v) < 0.05*abs(mean) and (max_v-min_v) > -0.05*abs(mean)):
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
    
    if (alpha <1 or beta<1):
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
  
redfield_current=[]
locallindblad_current=[]
realredfield_current=[]
universal_current=[]
elqme_current=[]
#dist_red_ule=[]
redfield_boundary=[]
locallindblad_boundary=[] 
realredfield_boundary=[]
universal_boundary=[]
elqme_boundary=[]
     

gvals=np.linspace(0.01,0.24,2)


for g in gvals:
        
    print("g value is ",g )
    b=50
    
    N=3
    w0max=0.8
    w0min=1.2
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
    
    
    delta=1
    mu1=mu
    mu2=mu
    beta1=1
    beta2=0.5
    
       
    
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
           # print(i,k,freq)
            if(np.absolute(freq) >=1/10**10):
                integral11[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta1,mu1,gamma1),limit=200,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
                integral12[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma1),limit=200,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0]  #left bath done
                integral21[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta2,mu2,gamma2),limit=200,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
                integral22[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma2),limit=200,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0]  #right bath
    
            if (np.absolute(freq)<=1/10**18):
                integral11[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func2,0,b,args=(tb,beta1,mu1,gamma1))[0]
                integral12[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath_2,0,b,args=(tb,gamma1))[0]
                integral21[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func2,0,b,args=(tb,beta2,mu2,gamma2))[0]
                integral22[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath_2,0,b,args=(tb,gamma2))[0]
    
    
    #            integral11[i,k]=0
    #            integral12[i,k]=0
    #            integral21[i,k]=0
    #            integral22[i,k]=0
            
    
            
            
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
       
    
    ########### Now we do realredfield shit #####################################################
    ## we basically have to replace the constants with their real parts.
    
    L_rr=spre(pre)+spost(post)
    
    for i in range(number):
        for k in range(number):
            vi=eigstates[i]
            vk=eigstates[k]
            
            op1=epsilon*epsilon*constant11[i,k].real*vi*vi.dag()*c_1*vk*vk.dag()*c_1.dag()
            op2=epsilon*epsilon*constant12[i,k].real*c_1.dag()*vi*vi.dag()*c_1*vk*vk.dag()
            
            op3=epsilon*epsilon*constant11[i,k].real*c_1.dag()
            op4=vi*vi.dag()*c_1*vk*vk.dag()
            op5=epsilon*epsilon*constant12[i,k].real*c_1.dag()
            
            
            L_rr=L_rr+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L_rr=L_rr+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
            
            op1=epsilon*epsilon*constant21[i,k].real*vi*vi.dag()*c_N*vk*vk.dag()*c_N.dag()
            op2=epsilon*epsilon*constant22[i,k].real*c_N.dag()*vi*vi.dag()*c_N*vk*vk.dag()
            
            op3=epsilon*epsilon*constant21[i,k].real*c_N.dag()
            op4=vi*vi.dag()*c_N*vk*vk.dag()
            op5=epsilon*epsilon*constant22[i,k].real*c_N.dag()
            
            
            L_rr=L_rr+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L_rr=L_rr+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
    
    
    ss_realredfield,dict_realredfield=steadystate(L_rr,return_info=return_info)
    L_redfield=L
        
    ########### ELQME shit ####################
    
    
    pre=-1.0j*H_S
    post=1.0j*H_S

    L=spre(pre)+spost(post)
    
    for i in range(number):
        for k in range(number):
            vi=eigstates[i]
            vk=eigstates[k]
            
            op1=epsilon*epsilon*constant11[i,k]*vi*vi.dag()*c_1*vk*vk.dag()*c_1.dag()*vi*vi.dag()
            op2=epsilon*epsilon*constant12[i,k]*vk*vk.dag()*c_1.dag()*vi*vi.dag()*c_1*vk*vk.dag()
            
            op3=epsilon*epsilon*constant11[i,k]*vk*vk.dag()*c_1.dag()*vi*vi.dag()
            op4=vi*vi.dag()*c_1*vk*vk.dag()
            op5=epsilon*epsilon*constant12[i,k]*vk*vk.dag()*c_1.dag()*vi*vi.dag()
            
            
            L=L+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L=L+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
            
            op1=epsilon*epsilon*constant21[i,k]*vi*vi.dag()*c_N*vk*vk.dag()*c_N.dag()*vi*vi.dag()
            op2=epsilon*epsilon*constant22[i,k]*vk*vk.dag()*c_N.dag()*vi*vi.dag()*c_N*vk*vk.dag()
            
            op3=epsilon*epsilon*constant21[i,k]*vk*vk.dag()*c_N.dag()*vi*vi.dag()
            op4=vi*vi.dag()*c_N*vk*vk.dag()
            op5=epsilon*epsilon*constant22[i,k]*vk*vk.dag()*c_N.dag()*vi*vi.dag()
            
            
            L=L+spre(-op2-op1.dag())+spost(-op1-op2.dag())
            L=L+spre(op3)*spost(op4)+spre(op4)*spost(op5)+spre(op4.dag())*spost(op3.dag()) +spre(op5.dag())*spost(op4.dag())
            
            
            
    #Variables needed for for iterative-lgmres to work. 
    return_info=True
    print('ELQME Liouvillian constructed, Computing steady-state ...')
    ss_elqme,dict_elqme=steadystate(L,return_info=return_info)
    
    L_elqme=L

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
    
    X_list=[]
    X_list.append(create_sigmax(N,1))
    X_list.append(create_sigmay(N,1))
    X_list.append(create_sigmax(N,N))
    X_list.append(create_sigmay(N,N))
    
    print('starting lambshift calculations.')
    lambshift=0
    ls_left=0
    ls_right=0
    # for alpha_index in range(1,5):
    #     for beta_index in range(1,5):
    #         for l in range(number):
    #             for m in range(number):
    #                 for n in range(number):
    #                     Em=eigenergies[m]
    #                     En=eigenergies[n]
    #                     El=eigenergies[l]
    #                     factor_real=-2*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(El-Em,En-El,alpha_index,beta_index,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,0),limit=1000,weight='cauchy',wvar=0)[0]
    #                     factor_imag=-2.0j*np.pi*epsilon*epsilon*integrate.quad(lamb_integrand,-b,b,args=(El-Em,En-El,alpha_index,beta_index,gamma1,gamma2,beta1,beta2,mu1,mu2,tb,1),limit=1000,weight='cauchy',wvar=0)[0]
    #                     term=(factor_real+factor_imag)*eigstates[m]*eigstates[m].dag()*X_list[alpha_index-1]*eigstates[l]*eigstates[l].dag()*X_list[beta_index-1]*eigstates[n]*eigstates[n].dag()
                        
                        
    #                     lambshift=lambshift+term
                        
    #                     if (alpha_index<=2 and beta_index<=2):
    #                         ls_left=ls_left+term
    #                     elif (alpha_index >=3 and beta_index >=3):
    #                         ls_right=ls_right+term
                        
        
        
        
    print('Lambshift for full ULE completed')
    
    
    ss_universal=steadystate(H_S+lambshift,Cops_universal)
    
    
    # ######### Trace Distance calculations #########################
    
    
    
    
    # dist_red_ule.append(tracedist(ss_redfield,ss_universal))
    
   # L_redfield=L
    L_universal=liouvillian(H_S+lambshift,Cops_universal)
    #L_rr is realredfield, L_elqme, L_redfield are already assigned
    L_lindblad=liouvillian(H_S,Cops)
    
    
    ### we will create the current operator and site 1 and append it to our data.
   
    
    
    vec_uni=[] #stores current as a function of position.
    vec_red=[]
    vec_realred=[]
    vec_lindblad=[]
    vec_elqme=[]
    
    
    #### WE will now check if current values are same accross sites :
    for pos in range(1,N):
        current_op=4.0j*g*(create_sm(N,pos).dag()*create_sm(N,pos+1)-create_sm(N,pos)*create_sm(N,pos+1).dag())
        vec_uni.append(expect(current_op,ss_universal))
        vec_red.append(expect(current_op,ss_redfield))
        vec_realred.append(expect(current_op,ss_realredfield))
        vec_lindblad.append(expect(current_op,ss_lindblad)) 
        vec_elqme.append(expect(current_op,ss_elqme))
        
    if (convergencecheck(vec_uni)==1):
        print('Current at different sites are not matching in ULE')
    if (convergencecheck(vec_red)==1):
        print('Current at different sites are not matching in redfield')
    if (convergencecheck(vec_realred)==1):
        print('Current at different sites are not matching in real redfield')
    if (convergencecheck(vec_lindblad)==1):
        print('Current at different sites are not matching in lindblad')
    if (convergencecheck(vec_lindblad)==1):
        print('Current at different sites are not matching in ELQME')
    
    universal_current.append(vec_uni)
    redfield_current.append(vec_red)
    realredfield_current.append(vec_realred)
    locallindblad_current.append(vec_lindblad)
    elqme_current.append(vec_elqme)
    
    ### WE Will check degeneracy now
    if (np.absolute(L_universal.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in ULE")
    if (np.absolute(L_redfield.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in redfield")
    if (np.absolute(L_rr.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in RealRedfield")
    if (np.absolute(L_lindblad.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in local lindblad")
    if (np.absolute(L_elqme.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in ELQME")


    ############ Now we compute boudary current
          
    Mz=create_magnetization(N)
    
    ########### local linblad
    ll_left_ls=(1.0j*ss_lindblad*commutator( (Deltadash1+0.5*Delta1)*create_sigmaz(N,1) , Mz  )).tr()
    ll_right_ls=(1.0j*ss_lindblad*commutator( (DeltadashN+0.5*DeltaN)*create_sigmaz(N,N) , Mz  )).tr()
    
    ll_left_lindblad=(ss_lindblad*(Cops[0].dag()*Mz*Cops[0]-0.5*Mz*Cops[0].dag()*Cops[0]-0.5*Cops[0].dag()*Cops[0]*Mz + Cops[1].dag()*Mz*Cops[1]-0.5*Mz*Cops[1].dag()*Cops[1]-0.5*Cops[1].dag()*Cops[1]*Mz )).tr()
    ll_right_lindblad=(ss_lindblad*(Cops[2].dag()*Mz*Cops[2]-0.5*Mz*Cops[2].dag()*Cops[2]-0.5*Cops[2].dag()*Cops[2]*Mz + Cops[3].dag()*Mz*Cops[3]-0.5*Mz*Cops[3].dag()*Cops[3]-0.5*Cops[3].dag()*Cops[3]*Mz )).tr()

    locallindblad_boundary.append([ll_left_ls+ll_left_lindblad,ll_right_ls+ll_right_lindblad])

    ############## universal lindblad ###########

    ule_left_ls=(1.0j*ss_universal*commutator( ls_left , Mz  )).tr()
    ule_right_ls=(1.0j*ss_universal*commutator( ls_right, Mz  )).tr()
    
    ule_left_lindblad=(ss_universal*(Cops_universal[0].dag()*Mz*Cops_universal[0]-0.5*Mz*Cops_universal[0].dag()*Cops_universal[0]-0.5*Cops_universal[0].dag()*Cops_universal[0]*Mz + Cops_universal[1].dag()*Mz*Cops_universal[1]-0.5*Mz*Cops_universal[1].dag()*Cops_universal[1]-0.5*Cops_universal[1].dag()*Cops_universal[1]*Mz )).tr()
    ule_right_lindblad= (ss_universal*(Cops_universal[2].dag()*Mz*Cops_universal[2]-0.5*Mz*Cops_universal[2].dag()*Cops_universal[2]-0.5*Cops_universal[2].dag()*Cops_universal[2]*Mz + Cops_universal[3].dag()*Mz*Cops_universal[3]-0.5*Mz*Cops_universal[3].dag()*Cops_universal[3]-0.5*Cops_universal[3].dag()*Cops_universal[3]*Mz )).tr()
    

    universal_boundary.append([ule_left_ls+ule_left_lindblad,ule_right_ls+ule_right_lindblad])
    
    
    ################ REdfield and realredfield 
    redfield_left=0
    redfield_right=0
    
    realredfield_left=0
    realredfield_right=0
    
    elqme_left=0
    elqme_right=0

    for index in indices1:
        i=index[0]
        k=index[1]
        redfield_left=redfield_left+commutator(ss_redfield*list1[i][k],c_1.dag())*constant11[i,k]
        redfield_left=redfield_left+commutator(c_1.dag(),list1[i][k]*ss_redfield)*constant12[i,k]
        realredfield_left=realredfield_left+commutator(ss_realredfield*list1[i][k],c_1.dag())*constant11[i,k].real
        realredfield_left=realredfield_left+commutator(c_1.dag(),list1[i][k]*ss_realredfield)*constant12[i,k].real
        elqme_left=elqme_left+commutator(ss_elqme*list1[i][k],list1[i][k].dag())*constant11[i,k]
        elqme_left=elqme_left+commutator(list1[i][k].dag(),list1[i][k]*ss_elqme)*constant12[i,k]
        
        
    for index in indices2:
        i=index[0]
        k=index[1]
        redfield_right=redfield_right+commutator(ss_redfield*list2[i][k],c_N.dag())*constant21[i,k]
        redfield_right=redfield_right+commutator(c_N.dag(),list2[i][k]*ss_redfield)*constant22[i,k]
        realredfield_right=realredfield_right+commutator(ss_realredfield*list2[i][k],c_N.dag())*constant21[i,k].real
        realredfield_right=realredfield_right+commutator(c_N.dag(),list2[i][k]*ss_realredfield)*constant22[i,k].real
        elqme_right=elqme_right+commutator(ss_elqme*list2[i][k],list2[i][k].dag())*constant21[i,k]
        elqme_right=elqme_right+commutator(list2[i][k].dag(),list2[i][k]*ss_elqme)*constant22[i,k]

    current_red_left=(-epsilon*epsilon*(redfield_left+redfield_left.dag())*Mz).tr()
    current_red_right=(-epsilon*epsilon*(redfield_right+redfield_right.dag())*Mz).tr()
    current_realred_left=(-epsilon*epsilon*(realredfield_left+realredfield_left.dag())*Mz).tr()
    current_realred_right=(-epsilon*epsilon*(realredfield_right+realredfield_right.dag())*Mz).tr()
    current_elqme_left=(-epsilon*epsilon*(elqme_left+elqme_left.dag())*Mz).tr()
    current_elqme_right=(-epsilon*epsilon*(elqme_right+elqme_right.dag())*Mz).tr()
    
    redfield_boundary.append([current_red_left,current_red_right])
    realredfield_boundary.append([current_realred_left,current_realred_right])
    elqme_boundary.append([current_elqme_left,current_elqme_right])














######################## now we store data. ################


string="N={},w0max={},w0min={},tb={},epsilon={},gamma1={},gamma2={}, delta={}, \n mu1={},mu2={},beta1={},beta2={}".format(N,w0max,w0min,tb,epsilon,gamma1,gamma2,delta,mu1,mu2,beta1,beta2)


#data=np.column_stack((gvals,redfield_current,realredfield_current,locallindblad_current))
#header_string=string+"\n GVALS || REDFIELD || REALREDFIELD || LOCALLINDBLAD "
#np.savetxt('N={},10delta={},current.txt'.format(N,10*delta),data,header=header_string)
y1=[]
y2=[]
i=0
for g in gvals:
   y1.append(np.abs(redfield_boundary[i][0]))
   y2.append(np.abs(universal_boundary[i][0]))
   i=i+1
   
plt.plot(gvals,y1,'bo-',label='Redfield')
plt.plot(gvals,y2,'rx-',label='ULE')
plt.ylabel('Current')
plt.xlabel('g')
plt.title(string)
plt.legend()
plt.savefig('blah.png',bbox_inches='tight',dpi=500)






