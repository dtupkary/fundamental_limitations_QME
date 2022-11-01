#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:18:27 2020
Code to play the 6 panel figure for the paper.
@author: devashish
"""


import numpy as np
import matplotlib.pyplot as plt





size1=4 #markersize
size2=2 #linewidth

size3=13 #font for title etc
size4=11





##### first we plot tracedistance vs g in the 11 plo##############t





data=np.loadtxt("./thermalplot/N=6,delta=10,thermalization.txt")

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
ax11.set_ylabel('Trace Distance to Thermal State',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('thermalplot',dpi=500,bbox_inches='tight')


################### ends ####################
################## Now we do beta plot #######################

data=np.loadtxt("./betaplot/N=6,delta=10,betathermal.txt")

dist_redfield=[y for _, y in sorted(zip(data[:,0],data[:,1])) ]
dist_realredfield=[y for _, y in sorted(zip(data[:,0],data[:,2]))  ]
dist_lindblad=[y for _, y in sorted(zip(data[:,0],data[:,3]))  ]
dist_universal=[y for _, y in sorted(zip(data[:,0],data[:,1]))  ]
betavals=sorted(data[:,0])

Tvals=[]

for beta in betavals:
    Tvals.append(1/beta)
  
    
l=len(betavals)
start=0

fig,ax=plt.subplots()
ax.plot(Tvals[start:l],dist_redfield[start:l],'rx-',label='RQE',markersize=size1,linewidth=size2)
ax.plot(Tvals[start:l],dist_realredfield[start:l],'k+-',label='RRQE',markersize=size1,linewidth=size2)
ax.plot(Tvals[start:l],dist_lindblad[start:l],'go-',label='LQE',markersize=size1,linewidth=size2)
ax.plot(Tvals[start:l],dist_universal[start:l],'b.-',label='ULE',markersize=size1,linewidth=size2)

#ax11.set_title('(a)',fontsize=size3)
ax.set_xlabel('Temperature',fontsize=size3)
ax.set_ylabel('Trace Distance to Thermal State',fontsize=size3)
ax.legend(loc='best',fontsize=size4)
ax.set_xscale('log')
ax.yaxis.get_offset_text().set_fontsize(size4)


plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('betaplot',dpi=500,bbox_inches='tight')


#################### Now we do time-dynamics plots #############################
########### normal temp first ########################
data=np.loadtxt("./time_eq/N=6,delta=10,beta=10,thermal_time.txt")

tlist=data[:,0]
dist_redfield=data[:,1]
dist_realredfield=data[:,2]
dist_lindblad=data[:,3]
dist_universal=data[:,4]


fig,ax11=plt.subplots()

ax11.plot(tlist,dist_redfield,'r-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_realredfield,'k-.',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_lindblad,'g--',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_universal,'b:',label='ULE',markersize=size1,linewidth=size2)

#ax11.set_title('(a)',fontsize=size3)
ax11.set_xlabel('Time',fontsize=size3)
ax11.set_ylabel('Trace Distance to Thermal State',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('timeplot-1',dpi=500,bbox_inches='tight')



############################ Now we do High temperature time temperature plot ########################################

data=np.loadtxt("./time_eq/N=6,delta=10,beta=1,thermal_time.txt")

tlist=data[:,0]
dist_redfield=data[:,1]
dist_realredfield=data[:,2]
dist_lindblad=data[:,3]
dist_universal=data[:,4]


fig,ax11=plt.subplots()

ax11.plot(tlist,dist_redfield,'r-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_realredfield,'k-.',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_lindblad,'g--',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(tlist,dist_universal,'b:',label='ULE',markersize=size1,linewidth=size2)

#ax11.set_title('(a)',fontsize=size3)
ax11.set_xlabel('Time',fontsize=size3)
ax11.set_ylabel('Trace Distance to Thermal State',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

fig.tight_layout()
plt.savefig('timeplot-2',dpi=500,bbox_inches='tight')


################### Now we do magnetization plot (normal temperature) #######################################

data=np.loadtxt("./mag_eq/N=6,delta=10,beta=10,profile.txt")

zlist=data[:,0]
profile_redfield=data[:,1]
profile_realredfield=data[:,2]
profile_lindblad=data[:,3]
profile_universal=data[:,4]
#profile_theory=data[:,5]

fig,ax11=plt.subplots()

ax11.plot(zlist,profile_redfield,'rx-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_realredfield,'k+-.',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_lindblad,'go--',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_universal,'b.:',label='ULE',markersize=size1,linewidth=size2)
#ax11.plot(zlist,profile_theory,'c^-',label='Thermal',markersize=1,linewdith=size2)

#ax11.set_title('(a)',fontsize=size3)

anchor=profile_redfield[0]
ax11.set_xlabel('Position',fontsize=size3)
ax11.set_ylabel('Magnetization',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)
plt.ylim(anchor-0.1,anchor+0.1)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('magplot-1',dpi=500,bbox_inches='tight')


######################### Now we do magnetization plot( high temperature) ######################



data=np.loadtxt("./mag_eq/N=6,delta=10,beta=1,profile.txt")

zlist=data[:,0]
profile_redfield=data[:,1]
profile_realredfield=data[:,2]
profile_lindblad=data[:,3]
profile_universal=data[:,4]
#profile_theory=data[:,5]

fig,ax11=plt.subplots()

ax11.plot(zlist,profile_redfield,'rx-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_realredfield,'k+-.',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_lindblad,'go--',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_universal,'b.:',label='ULE',markersize=size1,linewidth=size2)
#ax11.plot(zlist,profile_theory,'c^-',label='Thermal',markersize=1,linewdith=size2)

#ax11.set_title('(a)',fontsize=size3)
anchor=profile_redfield[0]
plt.ylim(anchor-0.1,anchor+0.1)

ax11.set_xlabel('Position',fontsize=size3)
ax11.set_ylabel('Magnetization',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('magplot-2',dpi=500,bbox_inches='tight')




########################### now we plot current ################



data=np.loadtxt("./currentplot/N=6,delta=10,current.txt")

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
ax11.set_ylabel('Current',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('currentplot',dpi=500,bbox_inches='tight')


##############non eq mag profile#####################



data=np.loadtxt("./mag_noneq/N=6,delta=10,profile(noneq).txt")

zlist=data[:,0]
profile_redfield=data[:,1]
profile_realredfield=data[:,2]
profile_lindblad=data[:,3]
profile_universal=data[:,4]
#profile_theory=data[:,5]

fig,ax11=plt.subplots()

ax11.plot(zlist,profile_redfield,'rx-',label='RQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_realredfield,'k+-.',label='RRQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_lindblad,'go--',label='LQE',markersize=size1,linewidth=size2)
ax11.plot(zlist,profile_universal,'b.:',label='ULE',markersize=size1,linewidth=size2)
#ax11.plot(zlist,profile_theory,'c^-',label='Thermal',markersize=1,linewdith=size2)

#ax11.set_title('(a)',fontsize=size3)
anchor=profile_redfield[0]
plt.ylim(anchor-0.1,anchor+0.1)

ax11.set_xlabel('Position',fontsize=size3)
ax11.set_ylabel('Magnetization',fontsize=size3)
ax11.legend(loc='best',fontsize=size4)

ax11.yaxis.get_offset_text().set_fontsize(size4)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
plt.savefig('magplot_noneq-1',dpi=500,bbox_inches='tight')

