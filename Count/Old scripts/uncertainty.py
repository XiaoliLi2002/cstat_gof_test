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

#collect data
scount=[0 for x in range(240)]
bcount=[0 for x in range(240)]
bscount=[0 for x in range(240)]
wl=[0 for x in range(240)]

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

def resample(source,n):
    arr=[0 for x in range(n)]
    for x in range(n):
        y=random.randint(0,len(source)-1)
        arr[x]=source[y]
    return arr



k=100
npCbstd=[0 for x in range(k)]
npCbmean=[0 for x in range(k)]
npCsstd=[0 for x in range(k)]
npCsmean=[0 for x in range(k)]

pCbstd=[0 for x in range(k)]
pCbmean=[0 for x in range(k)]
pCsstd=[0 for x in range(k)]
pCsmean=[0 for x in range(k)]

#nonparameter
def npbootstrap(k):
    B = 2000
    n=240
    C_bs=[0 for x in range(B)]
    C_bb=[0 for x in range(B)]
    mean_bs=[0 for x in range(B)]
    mean_bb=[0 for x in range(B)]
    mean_bbs=[0 for x in range(B)]

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
    npCbmean[k]=statistics.mean(C_bb)
    npCbstd[k]=statistics.stdev(C_bb)
    npCsmean[k]=statistics.mean(C_bs)
    npCsstd[k]=statistics.stdev(C_bs)



# parameter bootstrap
def pbootstrap(k):
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

    pCbmean[k]=statistics.mean(C_bb)
    pCbstd[k]=statistics.stdev(C_bb)
    pCsmean[k]=statistics.mean(C_bs)
    pCsstd[k]=statistics.stdev(C_bs)

for i in range(k):
    print(i)
    npbootstrap(i)
    pbootstrap(i)

#plot
'''
x=np.arange(statistics.mean(pCbmean)-3*statistics.stdev(pCbmean),statistics.mean(pCbmean)+3*statistics.stdev(pCbmean),0.01)
y=normfun(x,statistics.mean(pCbmean),statistics.stdev(pCbmean))
plt.plot(x,y)

plt.hist(pCbmean,bins=15,rwidth=0.9,density=True)
plt.title('mean of C-stat of background counts distribution, parameter')
plt.xlabel('mean')
plt.ylabel('Frequency')
print(statistics.mean(pCbmean))
print(statistics.stdev(pCbmean))
plt.show()

x=np.arange(statistics.mean(pCbstd)-3*statistics.stdev(pCbstd),statistics.mean(pCbstd)+3*statistics.stdev(pCbstd),0.01)
y=normfun(x,statistics.mean(pCbstd),statistics.stdev(pCbstd))
plt.plot(x,y)

plt.hist(pCbstd,bins=15,rwidth=0.9,density=True)
plt.title('standard deviation of C-stat of background counts distribution, parameter')
plt.xlabel('std')
plt.ylabel('Frequency')
print(statistics.mean(pCbstd))
print(statistics.stdev(pCbstd))
plt.show()

x=np.arange(statistics.mean(pCsmean)-3*statistics.stdev(pCsmean),statistics.mean(pCsmean)+3*statistics.stdev(pCsmean),0.01)
y=normfun(x,statistics.mean(pCsmean),statistics.stdev(pCsmean))
plt.plot(x,y)

plt.hist(pCsmean,bins=15,rwidth=0.9,density=True)
plt.title('mean of C-stat of source counts distribution, parameter')
plt.xlabel('mean')
plt.ylabel('Frequency')
print(statistics.mean(pCsmean))
print(statistics.stdev(pCsmean))
plt.show()

x=np.arange(statistics.mean(pCsstd)-3*statistics.stdev(pCsstd),statistics.mean(pCsstd)+3*statistics.stdev(pCsstd),0.01)
y=normfun(x,statistics.mean(pCsstd),statistics.stdev(pCsstd))
plt.plot(x,y)

plt.hist(pCsstd,bins=15,rwidth=0.9,density=True)
plt.title('standard deviation of C-stat of source counts distribution, parameter')
plt.xlabel('std')
plt.ylabel('Frequency')
print(statistics.mean(pCsstd))
print(statistics.stdev(pCsstd))
plt.show()

x=np.arange(statistics.mean(npCbmean)-3*statistics.stdev(npCbmean),statistics.mean(npCbmean)+3*statistics.stdev(npCbmean),0.01)
y=normfun(x,statistics.mean(npCbmean),statistics.stdev(npCbmean))
plt.plot(x,y)

plt.hist(npCbmean,bins=15,rwidth=0.9,density=True)
plt.title('mean of C-stat of background counts distribution, nonparameter')
plt.xlabel('mean')
plt.ylabel('Frequency')
print(statistics.mean(npCbmean))
print(statistics.stdev(npCbmean))
plt.show()

x=np.arange(statistics.mean(npCbstd)-3*statistics.stdev(npCbstd),statistics.mean(npCbstd)+3*statistics.stdev(npCbstd),0.01)
y=normfun(x,statistics.mean(npCbstd),statistics.stdev(npCbstd))
plt.plot(x,y)

plt.hist(npCbstd,bins=15,rwidth=0.9,density=True)
plt.title('standard deviation of C-stat of background counts distribution, nonparameter')
plt.xlabel('std')
plt.ylabel('Frequency')
print(statistics.mean(npCbstd))
print(statistics.stdev(npCbstd))
plt.show()

x=np.arange(statistics.mean(npCsmean)-3*statistics.stdev(npCsmean),statistics.mean(npCsmean)+3*statistics.stdev(npCsmean),0.01)
y=normfun(x,statistics.mean(npCsmean),statistics.stdev(npCsmean))
plt.plot(x,y)

plt.hist(npCsmean,bins=15,rwidth=0.9,density=True)
plt.title('mean of C-stat of source counts distribution, nonparameter')
plt.xlabel('mean')
plt.ylabel('Frequency')
print(statistics.mean(npCsmean))
print(statistics.stdev(npCsmean))
plt.show()

x=np.arange(statistics.mean(npCsstd)-3*statistics.stdev(npCsstd),statistics.mean(npCsstd)+3*statistics.stdev(npCsstd),0.01)
y=normfun(x,statistics.mean(npCsstd),statistics.stdev(npCsstd))
plt.plot(x,y)

plt.hist(npCsstd,bins=15,rwidth=0.9,density=True)
plt.title('standard deviation of C-stat of source counts distribution, nonparameter')
plt.xlabel('std')
plt.ylabel('Frequency')
print(statistics.mean(npCsstd))
print(statistics.stdev(npCsstd))
plt.show()
'''