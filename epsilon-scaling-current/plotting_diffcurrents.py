#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:18:27 2020
Code to play the 6 panel figure for the paper.
@author: devashish
"""


import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"



size1=3 #markersize
size2=1 #linewidth

size3=13 #font for title etc
size4=11





##### first we plot tracedistance vs g in the 11 plo##############t





data1=np.loadtxt("./epsilonscaling_g1.txt")
data2=np.loadtxt("./epsilonscaling_g15.txt")
data3=np.loadtxt("./epsilonscaling_g2.txt")




epsilonvals1=data1[:,0]
diff1=data1[:,1]

epsilonvals2=data2[:,0]
diff2=data2[:,1]

epsilonvals3=data3[:,0]
diff3=data3[:,1]

m1,b1=np.polyfit(np.log(epsilonvals1),np.log(diff1),1)
m2,b2=np.polyfit(np.log(epsilonvals2),np.log(diff2),1)
m3,b3=np.polyfit(np.log(epsilonvals2),np.log(diff2),1)

fig,ax11=plt.subplots()
plt.loglog(epsilonvals1,diff1,'b.-',label='g=0.1,Slope={:.3f}'.format(m1),markersize=size1,linewidth=size2)
plt.loglog(epsilonvals2,diff2,'r.-',label='g=0.15,Slope={:.3f}'.format(m2),markersize=size1,linewidth=size2)
plt.loglog(epsilonvals3,diff3,'g.-',label='g=0.2,Slope={:.3f}'.format(m3),markersize=size1,linewidth=size2)

ax11.set_xlabel(r'$\epsilon$',fontsize=size3)
ax11.set_ylabel(r'$I_{(1,2)}-I_{(2,3)}$',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

############## needs to be changed
# line1=r'$N=3, \omega_0^1=1, \omega_0^2=1.5, \omega_0^3=2, t_b=0.01 $ '+'\n'
# line2=r'$ \epsilon=0.1, \Delta=1, \mu_1=\mu_3=-0.5 $' +'\n'
# line3=r'$\gamma_1=\gamma_3=1, \beta_1=5, \beta_2=0.5 $'
# string=line1+line2+line3
#ax11.text(0.01,1e-8,string,fontsize=9)

line1=r'$\omega_0^1=1,\omega_0^2=1.5,\omega_0^3=2$ '+'\n'
line2=r'$\beta_1=5, \beta_N=0.5$'
string=line1+line2

ax11.text(0.01,4*1e-8,string,fontsize=10,bbox=dict(facecolor='none', edgecolor='black', pad=2.0))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

############








fig.tight_layout()
plt.savefig('diff_scaling_multiple',dpi=500,bbox_inches='tight')


