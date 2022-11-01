

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:56:50 2020
Computes and plots the diff in site Current vs epsilon for ULE. 
The remaining 3 approaches have zero differences.  
This code is optimized so that we don't have to compute the same matrices for every epsilon! EPsilon anyway appears
as a constant in most expressions.' We will only compute ULE
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
    max_val=np.max(array)
    min_val=np.min(array)
    mean=np.mean(array)
    
    if (max_val < 1e-10 and min_val > -1e-10):
        return 1
    
    elif ((max_val-min_val) < 0.05*abs(mean) and (max_val-min_val) > -0.05*abs(mean)):
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

#epsilonvals=np.linspace(0.01,0.30,5)
epsilonvals=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
#gvals=[0.289]

epsilonindex=0
N=3
diff=np.zeros((9,N-2),dtype=np.float64) #stores the successive differences in currents.




## first we do cals for epsilon=1
print('Doing computation for epsilon=1')
epsilon=1

b=50

   
g=0.2
gmin=g
gmax=g

w0list=[1,1.5,2]
glist=np.linspace(gmin,gmax,N-1)

tb=0.01

gamma1=1 #gamma1 is the coupling to left bath. It shows up in spectral bath function
gamma2=1    #gamma2 is the coupling to the right bath.    



beta=0.1
mu=-0.5


delta=1
mu1=mu
mu2=mu
beta1=5
beta2=0.5

   

H_S=create_hamiltonian(w0list,glist,delta,N)


c_N=create_sm(N,N)  # we couple the Nth spin to the bath
c_1=create_sm(N,1)



eigenergies,eigstates=H_S.eigenstates()

#print("eigenenergies are : ",eigenergies)

spectrum=max(eigenergies)-min(eigenergies)

   # print("max energy , min energy = ", max(eigenergies),min(eigenergies))

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

############# Now we do the lambshift stuff #########################


ss_universal=steadystate(H_S,Cops_universal)

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
    
print('Lambshift for full ULE completed - Iterating over epsilon now')


######### Now do the sum over epsilon


for epsilon in epsilonvals:
    Cops_1=epsilon*Cops_universal[0]
    Cops_2=epsilon*Cops_universal[1]
    Cops_3=epsilon*Cops_universal[2]
    Cops_4=epsilon*Cops_universal[3]
    ls=epsilon*epsilon*lambshift
    ss_universal=steadystate(H_S+ls,[Cops_1,Cops_2,Cops_3,Cops_4])
    ss_universal_withoutls=steadystate(H_S,[Cops_1,Cops_2,Cops_3,Cops_4])

    vec_uni=[] #stores current as a function of position.
    
    
    #### WE will not check if current values are same accross sites :
    for pos in range(1,N):
        current_op=4.0j*g*(create_sm(N,pos).dag()*create_sm(N,pos+1)-create_sm(N,pos)*create_sm(N,pos+1).dag())
        vec_uni.append(expect(current_op,ss_universal))
        
    print(vec_uni)
    
    
    ### WE Will check degeneracy now
    L_universal=liouvillian(H_S+lambshift,Cops_universal)
    if (np.absolute(L_universal.eigenenergies()[-2]) < 1/(10**10)):
        print("Degeneracy in ULE")
      

    for i in range(N-2):
        diff[epsilonindex,i]=abs(vec_uni[i+1]-vec_uni[i])
    
    epsilonindex=epsilonindex+1


string="N={},w1={},w02={},w03={},tb={},g={},gamma1={},gamma2={}, delta={}, \n mu1={},mu2={},beta1={},beta2={}".format(N,w0list[0],w0list[1],w0list[2],tb,g,gamma1,gamma2,delta,mu1,mu2,beta1,beta2)
data=np.column_stack((epsilonvals,diff[:,0]))
np.savetxt('epsilonscaling_g2.txt',data,header=string)