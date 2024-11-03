import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import pylatex
import scipy.optimize as opt
np.set_printoptions(threshold=np.inf)

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

def generate_s(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*math.exp(-theta2*i/n)
    return arr

def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

def bootstrap(B,s,k,l):
    C=[0 for x in range(B)]
    n=len(s)
    for i in range(B):
        x=poisson_data(s)

        y = 0
        for j in range(len(s)):
            y += j / n * x[j]
        if y != 0:
            y = y / np.sum(x)
        else:
            y = 0.5
        x2 = Newton(y)
        if x2 == 0:
            x1 = np.mean(x)
        else:
            x1 = np.mean(x) * x2 / (1 - math.e ** (-x2))

        xopt = opt.minimize(LLF, [x1, x2], bounds=([[1e-7, 1e5], [-1e5, 1e5]]))
        theta1_mle=xopt['x'][0]
        theta2_mle=xopt['x'][1]
        print(theta1_mle)
        print(theta2_mle)
        r=generate_s(n,theta1_mle,theta2_mle)
        for j in range(len(r)):
            if x[j]== 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
    E_Cmin[k][l] = np.mean(C)
    Var_Cmin[k][l] = statistics.stdev(C)**2

def LLF(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*(math.log(theta[0],math.e)-theta[1]*(i+1)/n)-theta[0]*math.e**(-theta[1]*(i+1)/n)
    return -value

n=100   #10, 50, 100, 200
B=500
E_Cmin=[[0 for x in range(100)],[0 for x in range(100)],[0 for x in range(100)]]
Var_Cmin=[[0 for x in range(100)],[0 for x in range(100)],[0 for x in range(100)]]
mu=[[ (1+x)*0.1 for x in range(100)], [ (1+x)*0.1*0.5/(1-math.e**(-0.5)) for x in range(100)], [ (1+x)*0.1*1/(1-math.e**(-1)) for x in range(100)]]
theta2=[0,  0.5,  1.0]


for k in range(1):
    for l in range(100):
        s=generate_s(n,mu[k+1][l],theta2[k+1])

        C = [0 for x in range(B)]
        for i in range(B):
            x = poisson_data(s)

            y = 0
            for j in range(len(s)):
                y += j / n * x[j]
            if y != 0:
                y = y / np.sum(x)
            else:
                y = 0.5
            x2 = Newton(y)
            if x2 == 0:
                x1 = np.mean(x)
            else:
                x1 = np.mean(x) * x2 / (1 - math.e ** (-x2))

            xopt = opt.minimize(LLF, [x1, x2], bounds=([[1e-7, 1e5], [-1e5, 1e5]]))
            theta1_mle = xopt['x'][0]
            theta2_mle = xopt['x'][1]
            r = generate_s(n, theta1_mle, theta2_mle)
            for j in range(n):
                if x[j] == 0:
                    C[i] += 2 * r[j]
                else:
                    C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
        E_Cmin[k][l] = np.mean(C)
        Var_Cmin[k][l] = statistics.stdev(C) ** 2
        print(E_Cmin[k][l])
        print(l)
print('E:')
print(E_Cmin)
print('Var:')
print(Var_Cmin)
Expectation1=pd.DataFrame(E_Cmin[0])
Variance1=pd.DataFrame(Var_Cmin[0])
#Expectation2=pd.DataFrame(E_Cmin[1])
#Variance2=pd.DataFrame(Var_Cmin[1])
#Expectation3=pd.DataFrame(E_Cmin[2])
#Variance3=pd.DataFrame(Var_Cmin[2])
Expectation1.to_excel('Expectation_theta2=0.5.xlsx')
Variance1.to_excel('Variance_theta2=0.5.xlsx')
#Expectation2.to_excel('Expectation_theta2=0.5.xlsx')
#Variance2.to_excel('Variance_theta2=0.5.xlsx')
#Expectation3.to_excel('Expectation_theta2=1.xlsx')
#Variance3.to_excel('Variance_theta2=1.xlsx')
