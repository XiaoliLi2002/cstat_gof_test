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
plt.figure(dpi=600)
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def chi2fun(x,n):
    pdf=1/(2**(n/2)*scipy.special.gamma(n/2))*x**(n/2-1)*math.e**(-x/2)
    return pdf

def generate_s(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*((1+(i+1)/n)**theta2)
    return arr

def generate_s_gamma(n,alpha,beta):
    return scipy.stats.gamma.rvs(alpha,loc=0,scale=1/beta,size=n)

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
    s=generate_s(n,theta[0],theta[1])
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
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
    print(E)
    return float(E)

def Var(beta,n):
    s = generate_s(n, beta[0], beta[1])
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0,0]
    for i in range(n):
        var+=kapa_2(s[i])
    print(var)
    return var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))


n=100
B=1000
X = np.mat([[1 for x in range(n)], [math.log(1+(x + 1) / n,math.e) for x in range(n)]]).T
I = np.mat([1., 1.]).T

theta1=0.5
theta2=1
alpha=0.25
beta=0.5
#nb_r=4

iters=1
for l in range(iters):
    theta1_hat=theta1
    theta2_hat=theta2 # null distribution

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
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    C_re = sorted(C)

    x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(statistics.mean(C)-3*statistics.stdev(C),statistics.mean(C)+3*statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y,color='green',label='Alg.2')

    x = np.arange(expectation([theta1,theta2],n)-3*math.sqrt(Var([theta1,theta2],n)), expectation([theta1,theta2],n)+3*math.sqrt(Var([theta1,theta2],n)), 0.05)
    y = normfun(x, expectation([theta1,theta2],n), math.sqrt(Var([theta1,theta2],n)))
    plt.plot(x, y, color='tomato',label='Alg.3')

    plt.hist(C, bins=15, rwidth=0.9, density=True,color='cornflowerblue',label='Alg.4')
    plt.title('Distribution of C-stat of different algorithms in model B',fontsize=12)
    plt.xlabel('C-stat',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(linestyle='--', alpha=0.5)
    # plt.show()
    plt.savefig('hist_powerlaw.pdf')

