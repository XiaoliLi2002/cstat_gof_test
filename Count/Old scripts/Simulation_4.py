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

B=500 # size of bootstrap set
n=50 # number of bins
theta2=[0,  0.5,  1.0] # mean of exp data

E_Cmin=[[0 for x in range(100)],[0 for x in range(100)],[0 for x in range(100)]]
Var_Cmin=[[0 for x in range(100)],[0 for x in range(100)],[0 for x in range(100)]]
mu1=[ (1+x)*0.2 for x in range(100)]
mu2=[ (1+x)*0.2*0.5/(1-math.e**(-0.5)) for x in range(100)]
mu3=[ (1+x)*0.2*1/(1-math.e**(-1)) for x in range(100)]

mu=[[ (1+x)*0.2 for x in range(100)], [ (1+x)*0.2*0.5/(1-math.e**(-0.5)) for x in range(100)], [ (1+x)*0.2*1/(1-math.e**(-1)) for x in range(100)]]
def generate_s(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*math.exp(-theta2*i/n)
    return arr




# generate data based on arr mu
def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

# parameter bootstrap

def bootstrap(B,s,k,l):
    C=[0 for x in range(B)]
    for i in range(B):
        sample=poisson_data(s)
        for j in range(len(s)):
            if sample[j]== 0:
                C[i] += 2 * s[j]
            else:
                C[i] += 2 * (s[j] - sample[j] * math.log(s[j], math.e) - sample[j] + sample[j] * math.log(sample[j], math.e))
    E_Cmin[k][l] = np.mean(C)
    Var_Cmin[k][l] = statistics.stdev(C)**2

for k in range(3):
    for l in range(100):
        s=generate_s(n,mu[k][l],theta2[k])
        bootstrap(B,s,k,l)
        print(l)
print('E:')
print(E_Cmin)
print('Var:')
print(Var_Cmin)