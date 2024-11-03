import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
from scipy.stats import nbinom
import scipy.optimize as opt
import scipy
np.set_printoptions(threshold=np.inf)

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

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

def LLF(mu):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*math.log(mu[0],math.e)-mu[0]
    return -value

def p_value_norm(x,mu,sigma):
    z=(x-mu)/sigma
    return scipy.stats.norm.sf(z)

def p_value_chi(x,df):
    return scipy.stats.chi2.sf(x,df)

def generate_s(n,mu):
    return [mu for i in range(n)]

max=100


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
    s=generate_s(n,beta[0])
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])

    E=(-0.5*I.T*X.T*Sigma*X*(X.T*V*X)**(-1)*I)[0]
    for i in range(n):
        E+=kapa_1(s[i])
    print(float(E))
    return float(E)

def Var(beta,n):
    s = generate_s(n, beta[0])
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0]
    for i in range(n):
        var+=kapa_2(s[i])
    print(math.sqrt(var))
    return var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))

n=10
B=1000
X = np.mat([[1 for x in range(n)]]).T
I = np.mat([1.]).T

mu=2

s=[mu for i in range(n)]
x = poisson_data(s)
xopt = opt.minimize(LLF, [1], bounds=([[1e-2, 30]]))
mu_hat = xopt['x'][0]

r = generate_s(n, mu_hat)
Cmin=0
for j in range(n):
    if x[j] == 0:
        Cmin += 2 * r[j]
    else:
        Cmin += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

print(mu_hat,Cmin)
data=pd.DataFrame(x)
#data.to_excel('Data.xlsx')

s=generate_s(n,mu_hat)

C = [0 for x in range(B)]
for i in range(B):
    x = poisson_data(s)
    xopt = opt.minimize(LLF, [mu_hat], bounds=([[1e-2, 30]]))
    mu_hat = xopt['x'][0]
    r = generate_s(n, mu_hat)
    for j in range(n):
        if x[j] == 0:
            C[i] += 2 * r[j]
        else:
            C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
print(statistics.mean(C),statistics.stdev(C),p_value_norm(Cmin,statistics.mean(C),statistics.stdev(C)))
C_re=sorted(C)
data=pd.DataFrame(C_re)
#data.to_excel('C_min.xlsx')
for i in range(B):
    if C_re[i] < Cmin:
        i += 1
    else:
        print((B - i) / B)
        break
print(p_value_chi(Cmin,n-1))
print(  theory_test(Cmin,[mu_hat],n)  )
