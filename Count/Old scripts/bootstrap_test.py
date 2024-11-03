import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import scipy
import scipy.optimize as opt
np.set_printoptions(threshold=np.inf)

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

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

def LLF(theta): #Exponential
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*(math.log(theta[0],math.e)-theta[1]*(i+1)/n)-theta[0]*math.e**(-theta[1]*(i+1)/n)
    return -value

def p_value_norm(x,mu,sigma):
    z=abs(x-mu)/sigma
    return 2*min(scipy.stats.norm.sf(abs(z)),1-scipy.stats.norm.sf(abs(z)))

n=100
B=1000


theta1=[0.5,1,2,10]
theta2=1
s=generate_s(n,theta1[3],theta2)

x = poisson_data(s)
xopt = opt.minimize(LLF, [10, 1], bounds=([[1e-7, 1e5], [-1e5, 1e5]]))
theta1_hat = xopt['x'][0]
theta2_hat= xopt['x'][1]
r = generate_s(n, theta1_hat, theta2_hat)
Cmin=0
for j in range(n):
    if x[j] == 0:
        Cmin += 2 * r[j]
    else:
        Cmin += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

print(theta1_hat,theta2_hat,Cmin)
data=pd.DataFrame(x)
data.to_excel('Data.xlsx')

s=generate_s(n,theta1_hat,theta2_hat)

C = [0 for x in range(B)]
for i in range(B):
    x = poisson_data(s)
    xopt = opt.minimize(LLF, [1, 1], bounds=([[1e-7, 1e5], [-1e5, 1e5]]))
    theta1_mle = xopt['x'][0]
    theta2_mle = xopt['x'][1]
    r = generate_s(n, theta1_mle, theta2_mle)
    for j in range(n):
        if x[j] == 0:
            C[i] += 2 * r[j]
        else:
            C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
print(statistics.mean(C))
print(statistics.stdev(C))
print(p_value_norm(Cmin,statistics.mean(C),statistics.stdev(C)))
data=pd.DataFrame(C)
data.to_excel('C_min.xlsx')