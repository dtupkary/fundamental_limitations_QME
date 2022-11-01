# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:56:50 2020
Computes and plots magnetization profile for all 4 approachs + thermal state.
Lamb shift IS included
@author: Devashish
"""




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
    max=np.max(array)
    min=np.min(array)
    mean=np.mean(array)
    

    #print(max,min,"mean =",mean)
    if ((max-min) < 0.05*abs(mean) and (max-min) > -0.05*abs(mean)):
        return 1
    
    return 0


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


lindblad_list=[]
redfield_list=[]
theory_list=[] #only required for equilibrium studies 
universal_list=[]
realredfield_list=[]

lindblad_profile=[]
redfield_profile=[]
theory_profile=[]
universal_profile=[]
realredfield_profile=[]





b=50
limit_value=1500
N=3
g=0.2
w0min=1
w0max=2
gmin=g
gmax=g

w0list=np.linspace(w0min,w0max,N)
glist=np.linspace(g,g,N-1)

tb=1
epsilon=0.1
gamma1=1 #gamma1 is the coupling to left bath. It shows up in spectral bath function
gamma2=1    #gamma2 is the coupling to the right bath.    



beta=1
mu=-0.5


delta=1
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


number=len(eigenergies)

integral11=np.empty((number,number),dtype=np.cdouble) #stores J * N integral for left bath
integral12=np.empty((number,number),dtype=np.cdouble) # stores J integral (just to check) for the left bath
integral21=np.empty((number,number),dtype=np.cdouble) #stores J*N integral for right bath
integral22=np.empty((number,number),dtype=np.cdouble)




for i in range(number):
    for k in range(number):
        freq=eigenergies[k]-eigenergies[i]
        #print('i={},k={},freq={}'.format(i,k,freq))
                    
        if(np.absolute(freq) >= 1/10**10):
            integral11[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta1,mu1,gamma1),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
            integral12[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma1),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #left bath done
            integral21[i,k]=(-1.0j/(2*np.pi))*integrate.quad(func1,0,b,args=(tb,beta2,mu2,gamma2),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0] #func 1
            integral22[i,k]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,0,b,args=(tb,gamma2),limit=limit_value,weight='cauchy',wvar=eigenergies[k]-eigenergies[i])[0]  #right bath

        if (np.abs(freq)<=1/10**10): #it goes here when i=j.
        #if (freq==0): 
        #print('Freq=0 for i={},k={},freq={}'.format(i,k,freq))
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


print('Redfield Liouvillian constructed, Computing steady-state ...')
ss_redfield,dict_redfield=steadystate(L,return_info=return_info)
redfield_list.append(ss_redfield)
   

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


print('RealRedfield Liouvillian constructed, computing steady-state ...')
ss_realredfield,dict_realredfield=steadystate(L_rr,return_info=return_info)
realredfield_list.append(realredfield_list)


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



print('Local-Lindblad liouvillian constructed, computing steady-state ...')
t3=time.time()
ss_lindblad,dict_lindblad=steadystate(H,Cops,return_info=return_info)
t4=time.time()

lindblad_list.append(ss_lindblad)
   

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

print('Lambshift for ULE computed, Liouvillian constructed, computing steady-state...')
ss_universal=steadystate(H_S+lambshift,Cops_universal)
universal_list.append(ss_universal)


################### THeory calcs ################################
mag=create_magnetization(N)
theory=(-beta*(H_S-0.5*mu*mag)).expm()
ss_theory=theory/theory.tr()



######### Trace Distance calculations #########################



L_redfield=L
L_universal=liouvillian(H_S,Cops_universal)
#L_rr is realredfield
L_lindblad=liouvillian(H_S,Cops)



### WE Will check degeneracy now
if (np.absolute(L_universal.eigenenergies()[-2]) < 1/(10**10)):
    print("Degeneracy in ULE")
if (np.absolute(L_redfield.eigenenergies()[-2]) < 1/(10**10)):
    print("Degeneracy in redfield")
if (np.absolute(L_rr.eigenenergies()[-2]) < 1/(10**10)):
    print("Degeneracy in RealRedfield")
if (np.absolute(L_lindblad.eigenenergies()[-2]) < 1/(10**10)):
    print("Degeneracy in local lindblad")


string="N={},w0min={},w0max={},,g={},tb={},epsilon={},gamma1={},gamma2={}, delta={}, \n mu1={},mu2={},beta1={},beta2={}".format(N,w0min,w0max,g,tb,epsilon,gamma1,gamma2,delta,mu1,mu2,beta1,beta2)


zpositions=np.linspace(1,N,N)

for z in zpositions:
    pos=int(z)
    redfield_profile.append(expect(ss_redfield,create_sigmaz(N,pos)))
    realredfield_profile.append(expect(ss_realredfield,create_sigmaz(N,pos)))
    lindblad_profile.append(expect(ss_lindblad,create_sigmaz(N,pos)))
    universal_profile.append(expect(ss_universal,create_sigmaz(N,pos)))
    theory_profile.append(expect(ss_theory,create_sigmaz(N,pos)))
    
    
data=np.column_stack((zpositions,redfield_profile,realredfield_profile,lindblad_profile,universal_profile,theory_profile))
np.savetxt('N={},10delta={},10beta={},profile.txt'.format(N,int(10*delta),int(10*beta)),data,header=string+"\n magnetization profile.: TLIST ||| REDFIELD ||| REALREDFIELD ||| LINDBLAD ||| UNIVERSAL ||| THEORY")
    


#    
#
#
#
#anchor=expect(ss_theory,create_sigmaz(N,pos))
#
#
#
#print("trace distance to theory is \n")
#print("Redfield :- ",tracedist(ss_theory,ss_redfield))
#print("realredfield :- ",tracedist(ss_theory,ss_realredfield))
#print("lindblad :- ",tracedist(ss_theory,ss_lindblad))
#print("ULE :- ",tracedist(ss_theory,ss_universal))
#
#
#
#
#
#
#fig, ax1=plt.subplots(1,1)
#ax1.set_xlabel("position")
#ax1.set_ylabel("magnetization")
#ax1.set_title(string,fontsize=8)
#ax1.plot(zpositions,universal_profile,'bo',label='ULE')
#ax1.plot(zpositions,redfield_profile,'r+',label='redfield')
#ax1.plot(zpositions,realredfield_profile,'gx',label='realredfield')
#ax1.plot(zpositions,lindblad_profile,'y.',label='local lindblad')
#ax1.plot(zpositions,theory_profile,'k.',label='thermal')
#
#ax1.set_ylim(np.real(anchor)-0.1,np.real(anchor)+0.1)
#ax1.legend()
#
##ax1.tick_params(axis='both', which='major', labelsize=7)
#
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
##plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.1)
#fig.tight_layout()
#
#plt.savefig('magnetizationplot.png',dpi=500,bbox_inches='tight')
