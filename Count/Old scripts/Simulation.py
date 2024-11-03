import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
import matplotlib as mpl
import pylatex
np.set_printoptions(threshold=np.inf)

b=2000 # size of bootstrap set
n=100 # number of bins
k=10 # repetition times
gamma=[0.1, 0.5, 1.2, 4, 10] # mean of exp data

#parameters
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

q1=1
q2=1.96

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def resample(source,n):
    arr=[0 for x in range(n)]
    for x in range(n):
        y=random.randint(0,len(source)-1)
        arr[x]=source[y]
    return arr

def exp_data(mu,n):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=random.expovariate(1/mu)
    return arr

def norm_data(mu,sigma,n):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=random.normalvariate(mu,sigma)
        if arr[i]<=0:
            arr[i]=0
    return arr

def uniform_data(a,b,n):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]= random.uniform(a,b)
    return arr

# generate data based on arr mu
def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

# nonparameter bootstrap
def npbootstrap(arr,B):
    n=len(arr)
    C=[0 for x in range(B)]
    for i in range(B):
        sample=resample(arr,n)
        mean=np.mean(sample)
        for x in sample:
            if x == 0:
                C[i] += 2 * mean
            else:
                C[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))
    E_C=np.mean(C)
    std_C=statistics.stdev(C)
    print("nonparameter:")
    print(E_C,std_C)

'''
    # plot
    x = np.arange(E_C - 3 * std_C, E_C + 3 * std_C, 0.01)
    y = normfun(x, E_C, std_C)
    plt.plot(x, y)

    plt.hist(C, bins=15, rwidth=0.9, density=True)
    plt.title(' C-stat of background counts distribution, nonparameter')
    plt.xlabel('C-stat')
    plt.ylabel('Rescaled Frequency')
    plt.show()
'''

# parameter bootstrap
def pbootstrap(arr,B):
    n=len(arr)
    C=[0 for x in range(B)]
    for i in range(B):
        sample=poisson.rvs(mu=np.mean(arr),size=n)
        mean=np.mean(sample)
        for x in sample:
            if x == 0:
                C[i] += 2 * mean
            else:
                C[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))
    E_C = np.mean(C)
    std_C = statistics.stdev(C)
    print("parameter:")
    print(E_C, std_C)

'''
    # plot
    x = np.arange(E_C - 3 * std_C, E_C + 3 * std_C, 0.01)
    y = normfun(x, E_C, std_C)
    plt.plot(x, y)

    plt.hist(C, bins=15, rwidth=0.9, density=True)
    plt.title(' C-stat of background counts distribution, parameter')
    plt.xlabel('C-stat')
    plt.ylabel('Rescaled Frequency')
    plt.show()
'''
# calculate using empirical formula
def calculate(arr):
    N=len(arr)
    mean = np.mean(arr)
    # calculate C_min
    C_stat = 0
    for x in arr:
        if x == 0:
            C_stat += 2 * mean
        else:
            C_stat += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))
    print(C_stat)
    # calculate C_crit
    E_C_min = N * (F + E * pow(math.e, -beta * mean) + pow(math.e, -alpha * mean) * (
                A + B * mean + C * pow(mean - D, 2))) - 1

    Var_C_min = 2 * (N - 1) * (1 + pow(math.e, -Valpha * mean) * (VA + VB * mean + VC * pow(mean, 2)))
    std_C_min=math.sqrt(Var_C_min)

    C_min_crit = E_C_min + q * std_C_min
    print("Empirical:")
    print(E_C_min,std_C_min)



dataset = poisson_data(uniform_data(3,5,100))
np.savetxt("data.txt",dataset)
print(np.mean(dataset))

'''
#plot dataset
bin=[x+1 for x in range(n)]
plt.figure()
plt.scatter(bin,dataset,c='blue',s=1,label='Data point')
plt.xticks(range(0,100,20))
plt.yticks(range(0,90,20))
plt.xlabel("Bins",fontdict={'size':16})
plt.ylabel("Counts",fontdict={'size':16})
plt.legend(loc='best')
plt.show()
'''

calculate(dataset)
pbootstrap(dataset,b)
npbootstrap(dataset,b)