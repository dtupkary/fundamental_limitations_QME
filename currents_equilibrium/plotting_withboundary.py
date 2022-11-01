# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:32:07 2021
code plots the site current vs epsilon plot for the three approaches + two site currents for ULE
Built for equilibrium studies..
@author: tupka
"""
import matplotlib.pyplot as plt
import numpy as np


                                                 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


size1=1 #markersize
size2=3 #linewidth

size3=13 #font for title etc
size4=11



data=np.loadtxt('epsilonscaling_currents_withboundary.txt')
epsilonvals=data[:,0]
redfield_current=np.abs(data[:,1])
locallindblad_current=np.abs(data[:,2])
ule_current_site1=np.abs(data[:,3])
ule_current_site2=np.abs(data[:,4])
ule_boundary=np.abs(data[:,5])

fig, ax1=plt.subplots(1,1)


################## this needs to be updated every time ############


# line1=r'$N=3, \omega_0^1=1, \omega_0^2=1.5, \omega_0^3=2, t_b=0.01 $ '+'\n'
# line2=r'$ \epsilon=0.1, \Delta=1, \mu_1=\mu_3=-0.5 $' +'\n'
# line3=r'$\gamma_1=\gamma_3=1, \beta_1=\beta_2=0.5 $'
# string=line1+line2+line3



line1=r'$\omega_0^1=1,\omega_0^2=1.5,\omega_0^3=2$ '+'\n'
line2=r'$\beta_1=0.5, \beta_N=0.5$'
string=line1+line2


ax1.text(1e-3,4*1.7e-5,string,fontsize=10,bbox=dict(facecolor='none', edgecolor='black', pad=2.0))

######################################


ax1.set_xlabel(r' $\epsilon$ ',fontsize=size3)
ax1.set_ylabel("|Current|",fontsize=size3)



m_redfield,b=np.polyfit(np.log(epsilonvals),np.log(redfield_current),1)
m_locallindblad,b=np.polyfit(np.log(epsilonvals),np.log(locallindblad_current),1)
m_ulesite1,b=np.polyfit(np.log(epsilonvals),np.log(ule_current_site1),1)
m_ulesite2,b=np.polyfit(np.log(epsilonvals),np.log(ule_current_site2),1)
m_uleboundary,b=np.polyfit(np.log(epsilonvals),np.log(ule_boundary),1)


plt.loglog(epsilonvals,redfield_current,'rx-',markersize=size2,linewidth=size1,label='RQE. Slope={:.3f}'.format(m_redfield))
plt.loglog(epsilonvals,locallindblad_current,'go-',markersize=size2,linewidth=size1,label='LLE. Slope={:.3f}'.format(m_locallindblad))
plt.loglog(epsilonvals,ule_current_site1,'k+-',markersize=size2,linewidth=size1,label='ULE-Bond-1. Slope={:.3f}'.format(m_ulesite1))
plt.loglog(epsilonvals,ule_current_site2,'b.-',markersize=size2,linewidth=size1,label='ULE-Bond-2. Slope={:.3f}'.format(m_ulesite2))
plt.loglog(epsilonvals,ule_boundary,'yv-',markersize=size2,linewidth=size1,label='ULE-Boundary. Slope={:.3f}'.format(m_uleboundary))
ax1.legend(loc=0,fontsize=size4)


#ax1.tick_params(axis='both', which='major', labelsize=7)

#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xscale('log')
#plt.yscale('log')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.1)
fig.tight_layout()





plt.savefig('current_equilibrium_withboundary.png',dpi=600,bbox_inches='tight')



