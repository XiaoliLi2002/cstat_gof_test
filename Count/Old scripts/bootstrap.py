import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
import random
import statistics
from scipy.stats import poisson
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylatex
np.set_printoptions(threshold=np.inf)

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

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


scount=[0 for x in range(240)]
bcount=[0 for x in range(240)]
bscount=[0 for x in range(240)]
wl=[0 for x in range(240)]

#collect data
i=0
for x in range(240):
    wl[i] = 19.5 + 0.0125*i
    i=i+1

i=14903
k=0
for x in range(240):
    scount[k]=s_counts[i]
    bcount[k]=b_counts[i]
    i-=1
    k+=1
hdu_list.close()

for x in range(240):
    bscount[x]=scount[x]-0.09992*bcount[x]


# nonparameter bootstrap
def resample(source,n):
    arr=[0 for x in range(n)]
    for x in range(n):
        y=random.randint(0,len(source)-1)
        arr[x]=source[y]
    return arr

B = 2000
n=240
C_bs=[0 for x in range(B)] #C_min of background counts
C_bb=[0 for x in range(B)] #C_min of source counts
mean_bs=[0 for x in range(B)]
mean_bb=[0 for x in range(B)]
mean_bbs=[0 for x in range(B)] #mean of background-subtracted counts

for i in range(B):
    sample=resample(scount,n)
    mean=np.mean(sample)
    mean_bs[i]=mean
    for x in sample:
        if x == 0:
            C_bs[i] += 2 * mean
        else:
            C_bs[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))

for i in range(B):
    sample=resample(bcount,n)
    mean=np.mean(sample)
    mean_bb[i]=mean
    for x in sample:
        if x == 0:
            C_bb[i] += 2 * mean
        else:
            C_bb[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))

for i in range(B):
    sample=resample(bscount,n)
    mean=np.mean(sample)
    mean_bbs[i]=mean

#plot
'''
x=np.arange(200,400,0.1)
y=normfun(x,statistics.mean(C_bb),statistics.stdev(C_bb))
plt.plot(x,y)

plt.hist(C_bb,bins=15,rwidth=0.9,density=True)
plt.title('C-stat of background counts distribution, nonparameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(C_bb))
print(statistics.stdev(C_bb))
plt.show()



x=np.arange(200,350,0.1)
y=normfun(x,statistics.mean(C_bs),statistics.stdev(C_bs))
plt.plot(x,y)

plt.hist(C_bs,bins=15,rwidth=0.9,density=True)
plt.title('C-stat of source counts distribution, nonparameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(C_bs))
print(statistics.stdev(C_bs))
plt.show()



x=np.arange(0.9,1.5,0.01)
y=normfun(x,statistics.mean(mean_bs),statistics.stdev(mean_bs))
plt.plot(x,y)

plt.hist(mean_bs,bins=15,rwidth=0.9,density=True)
plt.title('mean of source counts distribution, nonparameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(mean_bs))
print(statistics.stdev(mean_bs))
plt.show()



x=np.arange(3.5,4.5,0.01)
y=normfun(x,statistics.mean(mean_bb),statistics.stdev(mean_bb))
plt.plot(x,y)

plt.hist(mean_bb,bins=15,rwidth=0.9,density=True)
plt.title('mean of background counts distribution, nonparameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(mean_bb))
print(statistics.stdev(mean_bb))
plt.show()

x=np.arange(0.5,1.1,0.01)
y=normfun(x,statistics.mean(mean_bbs),statistics.stdev(mean_bbs))
plt.plot(x,y)

plt.hist(mean_bbs,bins=15,rwidth=0.9,density=True)
plt.title('mean of background counts distribution, nonparameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(mean_bbs))
print(statistics.stdev(mean_bbs))
plt.show()
'''


# parameter bootstrap
'''
meanb=np.mean(bcount)
means=np.mean(scount)

B = 2000
n=240
C_bs=[0 for x in range(B)]
C_bb=[0 for x in range(B)]
mean_bs=[0 for x in range(B)]
mean_bb=[0 for x in range(B)]


for i in range(B):
    sample=poisson.rvs(mu=means,size=n)
    mean=np.mean(sample)
    mean_bs[i]=mean
    for x in sample:
        if x == 0:
            C_bs[i] += 2 * mean
        else:
            C_bs[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))

for i in range(B):
    sample=poisson.rvs(mu=meanb,size=n)
    mean=np.mean(sample)
    mean_bb[i]=mean
    for x in sample:
        if x == 0:
            C_bb[i] += 2 * mean
        else:
            C_bb[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))



#plot
x=np.arange(200,325,0.1)
y=normfun(x,statistics.mean(C_bb),statistics.stdev(C_bb))
plt.plot(x,y)

plt.hist(C_bb,bins=15,rwidth=0.9,density=True)
plt.title('C-stat of background counts distribution, parameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(C_bb))
print(statistics.stdev(C_bb))
plt.show()



x=np.arange(200,350,0.1)
y=normfun(x,statistics.mean(C_bs),statistics.stdev(C_bs))
plt.plot(x,y)

plt.hist(C_bs,bins=15,rwidth=0.9,density=True)
plt.title('C-stat of source counts distribution, parameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(C_bs))
print(statistics.stdev(C_bs))
plt.show()



x=np.arange(0.9,1.5,0.01)
y=normfun(x,statistics.mean(mean_bs),statistics.stdev(mean_bs))
plt.plot(x,y)

plt.hist(mean_bs,bins=15,rwidth=0.9,density=True)
plt.title('mean of source counts distribution, parameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(mean_bs))
print(statistics.stdev(mean_bs))
plt.show()



x=np.arange(3.5,4.5,0.01)
y=normfun(x,statistics.mean(mean_bb),statistics.stdev(mean_bb))
plt.plot(x,y)

plt.hist(mean_bb,bins=15,rwidth=0.9,density=True)
plt.title('mean of background counts distribution, parameter')
plt.xlabel('C-stat')
plt.ylabel('Frequency')
print(statistics.mean(mean_bb))
print(statistics.stdev(mean_bb))
plt.show()
'''