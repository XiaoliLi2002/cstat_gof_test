import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
np.set_printoptions(threshold=np.inf)

N=240 #error!
A=-0.600716
B=-2.66890
C=-2.360850
D=0.514446
E=0.331258
F=1.017396
beta=0.484436
alpha=3.937691
VA=-0.59488
VB=-1.0919
VC=0.85073
Valpha=0.94111

hdu_list=fits.open("../data/hrcf20957N002_pha2.fits")
hdu=hdu_list[1]
hdu_header=hdu.header
data=hdu.data
names=hdu.data.names
source_counts=hdu.data["COUNTS"]
s_counts=source_counts[0]
background_counts=hdu.data["BACKGROUND_UP"]
b_counts=background_counts[0]
wavelength_all=hdu.data["BIN_LO"]
wavelength=wavelength_all[0]


scount=[0 for x in range(240)] #source counts
bcount=[0 for x in range(240)] #background counts
wl=[0 for x in range(240)] #wavelength

#collect data
for x in range(240):
    wl[x] = 19.5 + 0.0125*x

i=14903 #index
for x in range(240):
    scount[x]=s_counts[i]
    bcount[x]=b_counts[i]
    i-=1
hdu_list.close()


mean = np.mean(bcount)

#calculate C_min
C_stat = 0
for x in bcount:
    if x == 0:
        C_stat += 2*mean
    else:
        C_stat += 2*(mean-x*math.log(mean, math.e)-x+x*math.log(x,math.e))
print(C_stat)

#calculate C_crit
E_C_min=N*(F+E*pow(math.e,-beta*mean)+pow(math.e,-alpha*mean)*(A+B*mean+C*pow(mean-D,2)))-1

Var_C_min=2*(N-1)*(1+pow(math.e,-Valpha*mean)*(VA+VB*mean+VC*pow(mean,2)))

q=1.96
C_min_crit=E_C_min+q*math.sqrt(Var_C_min)
print(C_min_crit)
print(E_C_min)
print(math.sqrt(Var_C_min))

'''''
plt.figure()
plt.scatter(wl,scount,c='blue',s=1,label='Source')
plt.scatter(wl,bcount,c='red',s=1,label='Background')
plt.xticks(range(20,23,1))
plt.yticks(range(0,11,1))
plt.xlabel("Wavelength",fontdict={'size':16})
plt.ylabel("Counts",fontdict={'size':16})
plt.legend(loc='best')
plt.show()
'''''

