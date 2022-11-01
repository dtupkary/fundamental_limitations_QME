#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:18:27 2020
Code to create the thermal plot panel figure for the paper.
@author: devashish
"""


import numpy as np
import matplotlib.pyplot as plt





size1=4 #markersize
size2=2 #linewidth

size3=13 #font for title etc
size4=11





##### first we plot tracedistance vs g in the 11 plo##############t


data=np.loadtxt("./N=3,10delta=10_thermalization_ex1.txt")

### we need to sort the yintres based on their gvals. Otherwise the lines in plotting get messed up"
dist_redfield=[y for _, y in sorted(zip(data[:,0],data[:,1])) ]
dist_realredfield=[y for _, y in sorted(zip(data[:,0],data[:,2]))  ]
dist_lindblad=[y for _, y in sorted(zip(data[:,0],data[:,3]))  ]
dist_universal=[y for _, y in sorted(zip(data[:,0],data[:,1]))  ]
gvals=sorted(data[:,0])


fig,ax11=plt.subplots()

ax11.plot(gvals,dist_redfield,'rx-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(gvals,dist_realredfield,'k+-',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(gvals,dist_lindblad,'go-',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(gvals,dist_universal,'b.-',label='ULE',markersize=size1,linewidth=size2)

#ax11.set_title('(a)',fontsize=size3)
ax11.set_xlabel('g',fontsize=size3)
ax11.set_ylabel(r'Trace Distance to $\rho_{th}$',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)


############ THIS NEEDS TO CHANGE FOR EVERY PLOT

line1=r'$N=3, \omega_0^1=1, \omega_0^2=1.5, \omega_0^3=2, t_b=0.01 $ '+'\n'
line2=r'$ \epsilon=0.1, \Delta=1, \mu_1=\mu_3=-0.5 $' +'\n'
line3=r'$\gamma_1=\gamma_3=1, \beta_1=\beta_2=1 $'
string=line1+line2+line3
ax11.text(0.12,0.012,string,fontsize=9)

#############




plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('thermalplot',dpi=500,bbox_inches='tight')


################### ends ####################

