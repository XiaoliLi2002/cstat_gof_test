import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
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

def LLF(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*(math.log(theta[0],math.e)-theta[1]*(i+1)/n)-theta[0]*math.e**(-theta[1]*(i+1)/n)
    return -value

n=100
B=1000

theta1_hat=[0 for x in range(B)]
theta2_hat=[0 for x in range(B)]
theta1=[0.5,1,2,10,30]
theta2=1
s=generate_s(n,theta1[4],theta2)

for i in range(B):
    x = poisson_data(s)
    y = 0
    xopt = opt.minimize(LLF, [30, 1], bounds=([[1e-7, 1e5], [-1e5, 1e5]]))
    theta1_hat[i] = xopt['x'][0]
    theta2_hat[i] = xopt['x'][1]


x=np.arange(25,35,0.05)
y=normfun(x,statistics.mean(theta1_hat),statistics.stdev(theta1_hat))
plt.plot(x,y)

plt.hist(theta1_hat,bins=10,rwidth=0.9,density=True)
plt.title('Distribution of Theta_1, parameter bootstrap')
plt.xlabel('Theta_1')
plt.ylabel('Frequency')
print(statistics.mean(theta1_hat))
print(statistics.stdev(theta1_hat))
plt.show()

x=np.arange(0.75,1.25,0.01)
y=normfun(x,statistics.mean(theta2_hat),statistics.stdev(theta2_hat))
plt.plot(x,y)

plt.hist(theta2_hat,bins=10,rwidth=0.9,density=True)
plt.title('Distribution of Theta_2, parameter bootstrap')
plt.xlabel('Theta_2')
plt.ylabel('Frequency')
print(statistics.mean(theta2_hat))
print(statistics.stdev(theta2_hat))
plt.show()