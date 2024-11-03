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

def poisson_data_constant(mu,n):
    return poisson.rvs(mu=mu,size=n).tolist()

def nb_data(r,mu):
    arr = [0. for x in range(len(mu))]
    prob=[0. for x in range(len(mu))]
    for i in range(len(mu)):
        prob[i]=r/(r+mu[i])
    for i in range(len(mu)):
        arr[i] = nbinom.rvs(r,prob[i],size=1)[0]
    return arr

def p_value_norm(x,mu,sigma):
    z=(x-mu)/sigma
    return scipy.stats.norm.sf(z)

def p_value_chi(x,df):
    return scipy.stats.chi2.sf(x,df)

def generate_s_exp(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*math.exp(-theta2*i/n)
    return arr

def generate_s_constant(n,mu):
    return [mu for i in range(n)]

def generate_s_powerlaw(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*((1+(i+1)/n)**theta2)
    return arr

def generate_s_norm(n,mu,sigma):
    return np.random.normal(mu,sigma,n)

def generate_s_gamma(n,alpha,beta):
    return scipy.stats.gamma.rvs(alpha,loc=0,scale=1/beta,size=n)

max=30


def poisson_dis(mu,i):
    return mu**i/math.factorial(i)*math.e**(-mu)

def Sigma_diag(Q,n):
    return kapa_12-kapa_11*Q*kapa_03*n

def expectation(beta,n):
    mu=beta[0]
    #V = np.diag([mu for i in range(n)])
    #Q=X*X.T/(mu*n)
    sigma=Sigma_diag(1/(mu*n),n)
    #Sigma=np.diag([sigma for i in range(n)])
    E=-0.5*sigma/mu
    E+=kapa_1*n
    print(float(E))
    return float(E)

def Var(beta,n):
    mu=beta[0]
    #V = np.diag([mu for i in range(n)])
    #k_11=np.mat([kapa_11 for i in range(n)]).T
    #var=(-k_11.T*X*X.T*k_11)[0,0]/(mu*n)
    var=-kapa_11**2*n/mu
    var+=kapa_2*n
    print(math.sqrt(var))
    return var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))


n =  10000
B=1000
X = np.mat([[1 for x in range(n)]]).T
I = np.mat([1.]).T
mu=0.001

iters=100
reject1=[0 for x in range(iters)] #chi2
reject2=[0 for x in range(iters)] #CANB
reject3=[0 for x in range(iters)] #Highorder
reject4=[0 for x in range(iters)] #B

for l in range(iters):
    print(l)
    #s = generate_s_exp(n,5,3)
    #s = generate_s_powerlaw(n,5,-3)
    #s=generate_s_constant(n,0.01)
    #s = generate_s_gamma(n,0.25,math.sqrt(0.25))



    x = poisson_data_constant(mu,n)
    if x==[0 for i in range(n)]:
        continue
    mu_hat=np.mean(x)
    r = generate_s_constant(n, mu_hat)
    Cmin = 0
    for j in range(n):
        if x[j] == 0:
            Cmin += 2 * r[j]
        else:
            Cmin += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    print(Cmin)
    s = generate_s_constant(n, mu_hat)

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data_constant(mu_hat,n)
        mu_mle=np.mean(x)
        r = generate_s_constant(n, mu_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] +=r[j]
            else:
                C[i] +=r[j] - x[j] * math.log(r[j]/x[j], math.e) - x[j]
        C[i]=2*C[i] # 2 is here
    if p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C)) < 0.05:
        reject2[l] = 1

    tem = 0
    C_re = sorted(C)
    for i in range(B):
        if C_re[i] < Cmin:
            i += 1
            tem+=1
        else:
            print((B - i) / B)
            break
    if (1-tem/B) < 0.05:
        reject4[l] = 1

    if p_value_chi(Cmin, n - 1) < 0.05:
        reject1[l] = 1
        print("chi")

    x = 0
    for k in range(max):
        if k == 0:
            x -= 2 * (k - mu_hat) * poisson_dis(mu_hat, k)
        else:
            x += 2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) * poisson_dis(mu_hat, k)
    kapa_1=x

    k_1 = kapa_1
    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) ** 2 * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) ** 2 * poisson_dis(mu_hat, k)
    kapa_2=x

    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) * (k - mu_hat) * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) * (k - mu_hat) * poisson_dis(mu_hat, k)
    kapa_11=x

    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) * (k - mu_hat) ** 2 * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) * (k - mu_hat) ** 2 * poisson_dis(mu_hat, k)
    kapa_12=x

    kapa_03=mu_hat

    if theory_test(Cmin, [mu_hat], n) < 0.05:
        reject3[l] = 1

print(np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4))
