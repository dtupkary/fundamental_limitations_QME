# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:20:43 2021
Plotting equilibrium magnetization profiles..
@author: tupka
"""

import matplotlib.pyplot as plt
import numpy as np

                                                 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


size1=3 #markersize
size2=1 #linewidth

size3=13 #font for title etc
size4=11


data=np.loadtxt("N=3,10delta=10,10beta=10,profile_ex1.txt")

zlist=data[:,0]
profile_redfield=data[:,1]
profile_realredfield=data[:,2]
profile_lindblad=data[:,3]
profile_universal=data[:,4]
profile_theory=data[:,5]
#profile_theory=data[:,5]

fig,ax11=plt.subplots()

ax11.plot(zlist,profile_redfield,'rx-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_realredfield,'k+-',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_lindblad,'go-',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_universal,'b.-',label='ULE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_theory,'y1-',label='Thermal',markersize=size1,linewidth=size2)
#ax11.plot(zlist,profile_theory,'c^-',label='Thermal',markersize=1,linewdith=size2)

#ax11.set_title('(a)',fontsize=size3)
#anchor=profile_redfield[0]
#plt.ylim(anchor-0.1,anchor+0.1)

ax11.set_xlabel('Position',fontsize=size3)
ax11.set_ylabel('Local Magnetization',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

########## textbox for paramters
# line1=r'$N=3, \omega_0^1=1, \omega_0^2=1.5, \omega_0^3=2, t_b=0.01 $ '+'\n'
# line2=r'$ \epsilon=0.1, \Delta=1, \mu_1=\mu_3=-0.5 $' +'\n'
# line3=r'$\gamma_1=\gamma_3=1, \beta_1=\beta_2=1 $'
# string=line1+line2+line3

line1=r'$\omega_0^1=1,\omega_0^2=1.5,\omega_0^3=2$ '+'\n'
line2=r'$\beta_1=1, \beta_N=1$'
string=line1+line2

ax11.text(1.9,-0.8,string,fontsize=10,bbox=dict(facecolor='none', edgecolor='black', pad=2.0))




#############




plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('magplot_eq.png',dpi=600,bbox_inches='tight')

