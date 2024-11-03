import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import scipy.optimize as opt
import scipy
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

def p_value_norm(x,mu,sigma):
    z=abs(x-mu)/sigma
    return 2*min(scipy.stats.norm.sf(abs(z)),1-scipy.stats.norm.sf(abs(z)))

def p_value_chi(x,df):
    return 2*min(scipy.stats.chi2.sf(abs(x),df),1-scipy.stats.chi2.sf(abs(x),df))

n=5
B=1000

theta1=0.5
theta2=1

s=generate_s(n,theta1,theta2)

x = poisson_data(s)
x=[0,0,2,0,0]
xopt = opt.minimize(LLF, [1, -1], bounds=([[1e-2, 30], [-5, 5]]))
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
    xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-2, 30], [-5, 5]]))
    theta1_mle = xopt['x'][0]
    theta2_mle = xopt['x'][1]
    r = generate_s(n, theta1_mle, theta2_mle)
    for j in range(n):
        if x[j] == 0:
            C[i] += 2 * r[j]
        else:
            C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
print(statistics.mean(C),statistics.stdev(C),p_value_norm(Cmin,statistics.mean(C),statistics.stdev(C)))
C_re=sorted(C)
data=pd.DataFrame(C_re)
data.to_excel('C_min.xlsx')
for i in range(B):
    if C_re[i]<Cmin:
        i+=1
    else:
        print(2*min(i,B-i)/1000)
        break
print(p_value_chi(Cmin,n-2))

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
    s=generate_s(n,beta[0],beta[1])
    I = np.mat([1., 1.]).T
    X = np.mat([[1 for x in range(n)], [(x + 1) / n for x in range(n)]]).T
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])

    E=(-0.5*I.T*X.T*Sigma*X*(X.T*V*X)**(-1)*I)[0]
    for i in range(n):
        E+=kapa_1(s[i])
    print(float(E))
    return float(E)

def Var(beta,n):
    s = generate_s(n, beta[0], beta[1])
    X = np.mat([[1 for x in range(n)], [(x + 1) / n for x in range(n)]]).T
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0]
    for i in range(n):
        var+=kapa_2(s[i])
    print(math.sqrt(var))
    return var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))

print(  theory_test(Cmin,[theta1_hat,theta2_hat],n)  )
