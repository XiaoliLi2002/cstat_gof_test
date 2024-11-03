import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import scipy.optimize as opt
from scipy.stats import nbinom
import scipy
np.set_printoptions(threshold=np.inf)
plt.figure(dpi=300,figsize=(18,12))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
#plt.rcParams['axes.linewidth'] = 2.0

x=[10,20,30,50,75,100]
yALB1=[.068,.07,.067,.097,.103,.093]
yALB2=[0.073,0.069,0.054,0.068,0.052,0.045]
yALB3=[.062,.055,.051,.065,.052,.043]
yALB4=[.05,.046,.046,.058,.043,.039]
yALB5=[0.05,0.046,0.043,0.058,0.046,0.042]

yAMB1=[0.088,0.104,0.119,0.155,0.203,0.256]
yAMB2=[0.075,0.06,0.058,0.064,0.055,0.054]
yAMB3=[0.064,0.055,0.056,0.061,0.053,0.052]
yAMB4=[0.05,0.046,0.046,0.058,0.043,0.039]
yAMB5=[0.05,0.046,0.043,0.058,0.046,0.042]

yASB1=[0.008,0.008,0.011,0.006,0.007,0.007]
yASB2=[0.034,0.053,0.043,0.037,0.037,0.036]
yASB3=[0.038,0.057,0.058,0.059,0.05,0.048]
yASB4=[0.043,0.029,0.020,0.030,0.034,0.032]
yASB5=[0.030,0.035,0.029,0.032,0.030,0.035]

yALB6=[.045,.05,.043,.058,.048,.043]
yAMB6=[.054,.053,.048,.044,.040,.048]
yASB6=[.049,.049,.051,.049,.045,.047]

yBPL1=[0.07,0.06,0.11,0.14,0.12,0.19]
yBPM1=[0.09,0.06,.15,.13,.21,.16]
yBPS1=[.0,.0,0,0,0,0]
yBPL2=[.07,.04,0.1,.07,.06,.011]
yBPM2=[.07,.07,.09,.05,.09,.02]
yBPS2=[0,.02,.02,.03,.01,.01]
yBPL3=[.06,.04,.07,.07,.06,.11]
yBPM3=[.04,.03,.08,.04,.07,.02]
yBPS3=[.02,.04,.03,.02,.03,.01]
yBPL4=[.05,.04,.06,.07,.06,.1]
yBPM4=[.06,.03,.06,.03,.08,.02]
yBPS4=[.02,.02,.01,.01,.02,.01]
yBPL5=[.07,.03,.07,.05,.06,.09]
yBPM5=[.09,.04,.08,.06,.05,.02]
yBPS5=[.35,.1,.08,.01,.01,.01]
yBPL6=[.06,.03,.06,.07,.03,.07]
yBPM6=[.07,.04,.07,.03,.09,.01]
yBPS6=[.08,.03,.04,.03,.02,.03]

yBEL1=[.1,.07,.12,.16,.13,.16]
yBEM1=[.09,.12,.13,.14,.16,.24]
yBES1=[0,0,0,0,0,0]
yBEL2=[.07,.05,.06,.07,.06,.05]
yBEM2=[.01,.08,.09,.04,.07,.08]
yBES2=[.01,0,.02,0,.01,.03]
yBEL3=[.06,.03,.06,.07,.05,.03]
yBEM3=[.05,.07,.05,.02,.08,.06]
yBES3=[.02,.04,.03,.02,.02,.06]
yBEL4=[.05,.06,.06,.05,.02,.04]
yBEM4=[.05,.09,.07,.02,.05,.06]
yBES4=[.01,.02,0,0,.01,.01]
yBEL5=[.07,.04,.06,.04,.03,.02]
yBEM5=[.08,.07,.09,.02,.07,.07]
yBES5=[.09,.03,.04,.02,0,.02]
yBEL6=[.03,.04,.06,.04,.05,.02]
yBEM6=[.01,.01,.06,.03,.05,.03]
yBES6=[.06,.03,.03,.03,.05,.04]

yCUL1=[0.159,0.189,0.244,0.326,0.415,0.481]
yCUM1=[0.303,0.482,0.612,0.799,0.923,0.96]
yCUS1=[0.295,0.471,0.563,0.743,0.789,0.894]
yCUL2=[0.157,0.176,0.208,0.265,0.324,0.355]
yCUM2=[0.269,0.377,0.49,0.635,0.767,0.873]
yCUS2=[0.42,0.649,0.771,0.915,0.965,0.991]
yCUL3=[0.145,0.162,0.205,0.256,0.318,0.351]
yCUM3=[0.24,0.367,0.482,0.628,0.759,0.869]
yCUS3=[0.422,0.696,0.825,0.954,0.987,0.996]
yCUL4=[0.126,0.141,0.19,0.229,0.293,0.331]
yCUM4=[0.22,0.345,0.463,0.613,0.749,0.864]
yCUS4=[0.385,0.64,0.765,0.91,0.962,0.991]
yCUL5=[0.125,0.137,0.19,0.209,0.282,0.321]
yCUM5=[0.219,0.341,0.438,0.59,0.733,0.841]
yCUS5=[0.385,0.642,0.76,0.909,0.962,0.99]
yCUL6=[0.122,0.13,0.154,0.199,0.265,0.295]
yCUM6=[0.193,0.314,0.413,0.545,0.697,0.8]
yCUS6=[0.43,0.692,0.794,0.936,0.979,0.994]
yCU1=[0.369,0.612,0.74,	0.903,	0.982,	0.997]
yCU2=[0.418,0.64,	0.743,	0.896,	0.977,	0.992]
yCU3=[0.4,0.635,	0.741,	0.895,	0.975,	0.992]
yCU4=[0.38,0.613,	0.723,	0.887,	0.966,	0.991]
yCU5=[0.382,0.601,	0.707,	0.881,	0.966,	0.99]
yCU6=[0.349,0.572,	0.679,	0.855,	0.956,	0.981]

yPower1=[[0.08,0.07,0.15,0.11,0.19,0.23],
[0.02,0.01,0.02,0.01,0.01,0.04],
[0,0,0,0,0,0],
[0.1,0.06,0.09,0.07,0.15,0.11],
[0.1,0.14,0.13,0.16,0.19,0.26],
[0.01,0,0	,0	,0	,0],
[0.06	,0.06,	0.07	,0.08	,0.05	,0.08],
[0.1	,0.07	,0.11	,0.12	,0.17	,0.16],
[0.03	,0.05	,0.03	,0.07	,0.07	,0.05]]
yPower2=[[0.09	,0.05	,0.1	,0.07	,0.08	,0.03],
[0.11	,0.06	,0.05	,0.04	,0.04	,0.07],
[0.02	,0	,0.01	,0	,0	,0],
[0.09	,0.06	,0.09,	0.06	,0.08,	0.04],
[0.08	,0.05	,0.04	,0.03	,0.03,	0.07],
[0.02	,0.03	,0.03,	0.07	,0.05,	0.04],
[0.07,	0.09	,0.07,	0.09	,0.03	,0.06],
[0.09,	0.07	,0.05,	0.08,	0.07	,0.06],
[0.08,	0.07,	0.04,	0.1	,0.09	,0.03]
]
yPower3=[[0.04	,0.03	,0.07	,0.05,	0.07	,0.03],
[0.04	,0.04,	0.04,	0.04,	0.05	,0.04],
[0.02	,0.06,	0.05	,0.04,	0.06,	0.08],
[0.07,	0.06,	0.07	,0.04,	0.05	,0.04],
[0.05,	0.06	,0.04,	0.03,	0.05,	0.06],
[0.01	,0.05,	0.04	,0.07,	0.07,	0.04],
[0.06	,0.05,	0.07,	0.07,	0.03	,0.04],
[0.07,	0.04	,0.05,	0.07,	0.07	,0.04],
[0.07,	0.07,	0.05,	0.07,	0.08	,0.03]
]
yPower4=[[0.08,	0.05,	0.07,	0.07,	0.06,	0.03],
[0.08,	0.06,	0.04	,0.03	,0.03,	0.06],
[0.01,	0.01,	0.01,	0,	0	,0],
[0.06,	0.05,	0.07,	0.02,	0.08,	0.04],
[0.07	,0.05,	0.03,	0.05,	0.03	,0.06],
[0.02,	0.02	,0.03	,0.05,	0.05,	0.05],
[0.05,	0.03,	0.07,	0.07	,0.02,	0.02],
[0.07,	0.03	,0.06,	0.06,	0.08,	0.04],
[0.08	,0.07	,0.02	,0.05,	0.07,	0.01],
]
yPower5=[[0.06,	0.02,	0.09,	0.03,	0.06,	0.03],
[0.12	,0.05,	0.03,	0.02	,0.05,	0.04],
[0.31	,0.26,	0.11,	0	,0.01,	0.01],
[0.05,	0.06,	0.04,	0.03	,0.07,	0.04],
[0.08	,0.03,	0.05,	0.03,	0.04,	0.06],
[0.42,	0.14,	0.04,	0.08,	0.05,	0.05],
[0.04,	0.05	,0.06	,0.06,	0.05	,0.03],
[0.06	,0.05	,0.06	,0.07,	0.06,	0.03],
[0.24,	0.1,	0.07,	0.06,	0.07,	0.03]
]
yPower6=[[0.03,	0.03,	0.04,	0.02,	0.05,	0.02],
[0.09	,0.05,	0.07,	0.07,	0.05,	0.07],
[0.05,	0.08,	0.08	,0.05	,0.05	,0.09],
[0.06,	0.05,	0.06,	0.03	,0.05	,0.04],
[0.07,	0.06	,0.03,	0.04,	0.05	,0.05],
[0.05,	0.04,	0.03,	0.06,	0.08,	0.04],
[0.04,	0.04,	0.09,	0.06,	0.03,	0.03],
[0.05,	0.02,	0.03,	0.06	,0.05	,0.05],
[0.07,	0.07,	0.03	,0.05	,0.06,	0.06]
]


xtick=[0,20,40,60,80,100]

plt.subplot(3,3,1)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yALB1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yALB2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yALB3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yALB4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yALB5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yALB6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model A, $\mu=5$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,2)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yAMB1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yAMB2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yAMB3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yAMB4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yAMB5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yAMB6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model A, $\mu=2$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,3)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yASB1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yASB2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yASB3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yASB4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yASB5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yASB6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model A, $\mu=0.5$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,4)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBPL1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBPL2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBPL3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBPL4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBPL5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBPL6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=5$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,5)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBPM1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBPM2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBPM3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBPM4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBPM5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBPM6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=2$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,6)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBPS1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBPS2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBPS3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBPS4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBPS5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBPS6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=0.5$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.legend(bbox_to_anchor=(1.0, 0.9))

plt.subplot(3,3,7)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBEL1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBEL2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBEL3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBEL4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBEL5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBEL6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Exponential, $\mu=5$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,8)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBEM1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBEM2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBEM3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBEM4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBEM5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBEM6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Exponential, $\mu=2$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,9)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yBES1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yBES2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yBES3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yBES4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yBES5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yBES6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Exponential, $\mu=0.5$, $k=1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)


plt.tight_layout()
plt.savefig('1.pdf')
plt.clf()



plt.subplot(3,3,1)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[0],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[0],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[0],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[0],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[0],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[0],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=5$, $k=3$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,2)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[1],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[1],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[1],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[1],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[1],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[1],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=2$, $k=3$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,3)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[2],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[2],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[2],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[2],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[2],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[2],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=0.5$, $k=3$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)


plt.subplot(3,3,4)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[3],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[3],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[3],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[3],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[3],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[3],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=5$, $k=0.1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,5)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[4],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[4],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[4],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[4],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[4],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[4],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=2$, $k=0.1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,6)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[5],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[5],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[5],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[5],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[5],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[5],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=0.5$, $k=0.1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.legend(bbox_to_anchor=(1.0, 0.9))

plt.subplot(3,3,7)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[6],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[6],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[6],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[6],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[6],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[6],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=5$, $k=-1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Type I Error',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,8)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[7],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[7],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[7],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[7],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[7],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[7],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=2$, $k=-1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(3,3,9)
plt.axhline(0.05,alpha=0.8,color='k',linestyle='--')
plt.plot(x,yPower1[8],color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yPower2[8],color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yPower3[8],color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yPower4[8],color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yPower5[8],color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yPower6[8],color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Powerlaw, $\mu=0.5$, $k=-1$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.savefig('2.pdf')
plt.clf()

plt.figure(dpi=300,figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(x,yCUL1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yCUL2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yCUL3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yCUL4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yCUL5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yCUL6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model C, $\alpha=25$, $\beta=\sqrt{\alpha}$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Power',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(2,2,2)
plt.plot(x,yCUM1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yCUM2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yCUM3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yCUM4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yCUM5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yCUM6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model C, $\alpha=4$, $\beta=\sqrt{\alpha}$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.legend(bbox_to_anchor=(1.0, 0.9))

plt.subplot(2,2,3)
plt.plot(x,yCU1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yCU2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yCU3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yCU4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yCU5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yCU6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model C, $\alpha=1$, $\beta=\sqrt{\alpha}$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel('Power',fontsize=18)
plt.xticks(xtick,fontsize=18)

plt.subplot(2,2,4)
plt.plot(x,yCUS1,color='orange', label='Alg.1',linestyle='--',marker='*')
plt.plot(x,yCUS2,color='green',label='Alg.2b',linestyle='--',marker='s')
plt.plot(x,yCUS3,color='tomato',label='Alg.3b',linestyle='--',marker='o')
plt.plot(x,yCUS4,color='cornflowerblue',label='Alg.4a',linestyle='--',marker='D')
#plt.plot(x,yCUS5,color='green',label='Alg.4b',linestyle='--',marker='^')
plt.plot(x,yCUS6,color='purple',label='Alg.4c',linestyle='--',marker='p')

plt.title(r'Model C, $\alpha=0.25$, $\beta=\sqrt{\alpha}$',fontsize=18)
plt.xlabel(r'$n$',fontsize=18)
plt.xticks(xtick,fontsize=18)
plt.tight_layout()
plt.savefig('3.pdf')
plt.clf()