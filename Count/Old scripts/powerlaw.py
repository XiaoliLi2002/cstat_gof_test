import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import scipy.optimize as opt
from scipy.stats import nbinom
import scipy
np.set_printoptions(threshold=np.inf)

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def generate_s(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*((1+(i+1)/n)**theta2)
    return arr

def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

def nb_data(r,mu):
    arr = [0. for x in range(len(mu))]
    prob=[0. for x in range(len(mu))]
    for i in range(len(mu)):
        prob[i]=r/(r+mu[i])
    for i in range(len(mu)):
        arr[i] = nbinom.rvs(r,prob[i],size=1)[0]
    return arr

def LLF(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*math.log(theta[0]*((1+(i+1)/n)**theta[1]),math.e)-theta[0]*((1+(i+1)/n)**theta[1])
    return -value

def p_value_norm(x,mu,sigma):
    z=(x-mu)/sigma
    return scipy.stats.norm.sf(z)

def p_value_chi(x,df):
    return scipy.stats.chi2.sf(x,df)

max=30

def poisson_dis(mu,i):
    return mu**i/math.factorial(i)*math.e**(-mu)

def kapa_1(mu):
    x=0
    for k in range(max):
        if k==0:
            x-=2*(k-mu)*poisson_dis(mu,k)
        else:
            x+=2*(k*math.log(k/mu,math.e)-k+mu)*poisson_dis(mu,k)
    return x

def kapa_2(mu):
    k_1=kapa_1(mu)
    x=0
    for k in range(max):
        if k==0:
            x+=(-2*(k-mu)-k_1)**2*poisson_dis(mu,k)
        else:
            x+=(2*(k*math.log(k/mu,math.e)-k+mu)-k_1)**2*poisson_dis(mu,k)

    return x

def kapa_11(mu):
    k_1=kapa_1(mu)
    x=0
    for k in range(max):
        if k==0:
            x+=(-2*(k-mu)-k_1)*(k-mu)*poisson_dis(mu,k)
        else:
            x+=(2*(k*math.log(k/mu,math.e)-k+mu)-k_1)*(k-mu)*poisson_dis(mu,k)
    return x

def kapa_12(mu):
    k_1 = kapa_1(mu)
    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu) - k_1) * (k - mu)**2 * poisson_dis(mu, k)
        else:
            x += (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu)**2 * poisson_dis(mu, k)
    return x

def kapa_03(mu):
    return mu

def Sigma_diag(s,i,Q,n):
    x=kapa_12(s[i])
    for j in range(n):
        x-=kapa_11(s[j])*Q[j,i]*kapa_03(s[i])
    return x

def expectation(beta,n):
    s=generate_s(n,beta[0],beta[1])
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])
    E=(-0.5*I.T*X.T*Sigma*X*(X.T*V*X)**(-1)*I)[0,0]
    for i in range(n):
        E+=kapa_1(s[i])
    return float(E)

def Var(beta,n):
    s = generate_s(n, beta[0], beta[1])
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0,0]
    for i in range(n):
        var+=kapa_2(s[i])
    return var

def theory_test(Cmin,beta,n):
    return p_value_norm(Cmin,expectation(beta,n),math.sqrt(Var(beta,n)))


n=100
B=1000
B1=1000
B2=1000
X = np.mat([[1 for x in range(n)], [math.log(1+(x + 1) / n,math.e) for x in range(n)]]).T
I = np.mat([1., 1.]).T

theta1=0.5
theta2=1

iters=1000
reject1=[0 for x in range(iters)] #chi2
reject2=[0 for x in range(iters)] #CANB
reject3=[0 for x in range(iters)] #Highorder
reject4=[0 for x in range(iters)] #B
reject5=[0 for x in range(iters)] #double B

for l in range(iters):
    print(l)
    s = generate_s(n, theta1, theta2)
    x = poisson_data(s)
    if x==[0 for i in range(n)]:
        continue
    xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, 30], [-50, 10]]))
    theta1_hat = xopt['x'][0]
    theta2_hat = xopt['x'][1]
    r = generate_s(n, theta1_hat, theta2_hat)
    Cmin = 0
    for j in range(n):
        if x[j] == 0:
            Cmin += r[j]
        else:
            Cmin += r[j] - x[j] * math.log(r[j] / x[j], math.e) - x[j]
    Cmin = 2 * Cmin

    s = generate_s(n, theta1_hat, theta2_hat)

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        if x==[0 for y in range(n)]:
            continue
        xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, 30], [-50, 10]]))
        theta1_mle = xopt['x'][0]
        theta2_mle = xopt['x'][1]
        r = generate_s(n, theta1_mle, theta2_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] +=r[j]
            else:
                C[i] +=(r[j] - x[j] * math.log(r[j]/x[j], math.e) - x[j])
        C[i]=2*C[i]
    if p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C)) < 0.05:
        reject2[l] = 1

    k = 0
    for i in range(B):
        if C[i] >= Cmin:
            k += 1
    if (1 - k / B) < 0.05:
        reject4[l] = 1

    if p_value_chi(Cmin, n - 2) < 0.05:
        reject1[l] = 1

    if theory_test(Cmin, [theta1_hat,theta2_hat], n) < 0.05:
        reject3[l] = 1

    #double bootstrap


print(np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4),np.mean(reject5))