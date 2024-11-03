import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import pylatex
np.set_printoptions(threshold=np.inf)

B=2000 # size of bootstrap set
n=[5, 10, 20, 30, 40, 50, 75, 100, 150, 300, 500, 750, 1000, 1500, 2000] # number of bins
gamma=[0.5, 1.0, 3, 5, 10] # mean of exp data




def exp_data(mu,n):
    arr=[0 for x in range(n)]
    if mu==0:
        return arr
    for i in range(n):
        arr[i]=random.expovariate(1/mu)
    return arr

# generate data based on arr mu
def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

# parameter bootstrap

def bootstrap(mu,B,n):
    C=[0 for x in range(B)]
    for i in range(B):
        sample=poisson_data(exp_data(mu,n))
        mean=np.mean(sample)
        for x in sample:
            if x == 0:
                C[i] += 2 * mean
            else:
                C[i] += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))
    E_C = np.mean(C)
    std_C = statistics.stdev(C)
    print("parameter:")
    print(E_C-1.96*std_C,E_C+1.96*std_C)

#left,right=chi2.interval(0.95,df=n[14]-1)
#print(left,right)

#for i in range(5):
 #   for j in range(15):
  #      bootstrap(gamma[i],B,n[j])
dataset=poisson_data([1 for i in range(100000)])
mean=np.mean(dataset)
print(np.mean(dataset))
C=0
for x in dataset:
    if x == 0:
        C += 2 * mean
    else:
        C += 2 * (mean - x * math.log(mean, math.e) - x + x * math.log(x, math.e))
print(C/100000)