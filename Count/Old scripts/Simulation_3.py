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
theta1=[0.2, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0]
theta2=[0,  0.5,  1.0] # mean of exp data

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

# solve function
def f(x):
    return 1/x-1/(math.e**x-1)

def g(x):
    return -1/(x**2)+math.e**x/(math.e**x-1)**2

def Newton(y):
    e=1e-7
    if y==0.5:
        return 0
    if y>0.5:
        x0=-1
        while abs(f(x0)-y)>e:
            x1=x0-(f(x0)-y)/g(x0)
            x0=x1
        return x0
    if y<0.5:
        x0=1
        while abs(f(x0)-y)>e:
            x1=x0-(f(x0)-y)/g(x0)
            x0=x1
        return x0


# parameter bootstrap

def bootstrap(B,s):
    C=[0 for x in range(B)]
    n=len(s)
    for i in range(B):
        sample=poisson_data(s)
        y=0
        for j in range(len(s)):
            y+=j/n*sample[j]
        if y!=0:
            y=y/np.sum(sample)
        else:
            y=0.5
        x2=Newton(y)
        if x2==0:
            x1=np.mean(sample)
        else:
            x1=np.mean(sample)*x2/(1-math.e**(-x2))
        t=generate_s(n,x1,x2)
        for j in range(len(t)):
            if sample[j]== 0:
                C[i] += 2 * t[j]
            else:
                C[i] += 2 * (t[j] - sample[j] * math.log(t[j], math.e) - sample[j] + sample[j] * math.log(sample[j], math.e))
    E_C = np.mean(C)
    std_C = statistics.stdev(C)
    print("parameter:")
    print(E_C-1.96*std_C,E_C+1.96*std_C, E_C)


x=10
left,right=chi2.interval(0.68,df=x-2)
print(left,right)


'''''
print('theta2=0:')
for i in range(7):
    print(i)
    for j in range(15):
        s=generate_s(n[j],theta1[i],theta2[0])
        bootstrap(B,s)


print('theta2=0.5:')
for i in range(7):
    print(i)
    for j in range(15):
        s=generate_s(n[j],theta1[i],theta2[1])
        bootstrap(B,s)


print('theta2=1.0:')
for i in range(7):
    print(i)
    for j in range(15):
        s=generate_s(n[j],theta1[i],theta2[2])
        bootstrap(B,s)
'''''
