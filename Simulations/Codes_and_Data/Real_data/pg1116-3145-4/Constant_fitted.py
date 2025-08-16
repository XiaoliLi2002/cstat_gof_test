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

def poisson_dis(mu,i):
    return mu**i/math.factorial(i)*math.e**(-mu)

#def Sigma_diag(Q,n):
    #return kapa_12-kapa_11*Q*kapa_03*n

def expectation(beta,n,kapa_1,kapa_2,kapa_11,kapa_12,kapa_03,X,I):
    mu=beta[0]
    #V = np.diag([mu for i in range(n)])
    #Q=X*X.T/(mu*n)
    #sigma=Sigma_diag(1/(mu*n),n)
    sigma = kapa_12-kapa_11*kapa_03/mu
    #Sigma=np.diag([sigma for i in range(n)])
    E=-0.5*sigma/mu
    E+=kapa_1*n
    #print(float(E))
    return float(E)

def Var(beta,n,kapa_1,kapa_2,kapa_11,kapa_12,kapa_03,X,I):
    mu=beta[0]
    #V = np.diag([mu for i in range(n)])
    #k_11=np.mat([kapa_11 for i in range(n)]).T
    #var=(-k_11.T*X*X.T*k_11)[0,0]/(mu*n)
    var=-kapa_11**2*n/mu
    var+=kapa_2*n
    #print(var)
    return var

def kapa1(mu,max):
    x = 0
    for k in range(max):
        if k == 0:
            x -= 2 * (k - mu) * poisson_dis(mu, k)
        else:
            x += 2 * (k * math.log(k / mu, math.e) - k + mu) * poisson_dis(mu, k)
    return x

def kapa2(mu,max):
    x = 0
    k_1=kapa1(mu,max)
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu) - k_1) ** 2 * poisson_dis(mu, k)
        else:
            x += (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) ** 2 * poisson_dis(mu, k)
    return x

def kapa11(mu,max):
    x = 0
    k_1 = kapa1(mu, max)
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu) - k_1) * (k - mu) * poisson_dis(mu, k)
        else:
            x += (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu) * poisson_dis(mu, k)
    return x

def kapa12(mu,max):
    x = 0
    k_1 = kapa1(mu, max)
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu) - k_1) * (k - mu) ** 2 * poisson_dis(mu, k)
        else:
            x += (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu) ** 2 * poisson_dis(mu,k)
    return x

def kapa03(mu):
    return mu

def theory_test(Cmin,beta,n,X,I,max=30):
    mu_hat=beta[0]

    kapa_1=kapa1(mu_hat,max)

    kapa_2 =kapa2(mu_hat,max)

    kapa_11 = kapa11(mu_hat,max)

    kapa_12 = kapa12(mu_hat,max)

    kapa_03 = kapa03(mu_hat)
    mean=expectation(beta,n,kapa_1,kapa_2,kapa_11,kapa_12,kapa_03,X,I)
    std=math.sqrt(Var(beta,n,kapa_1,kapa_2,kapa_11,kapa_12,kapa_03,X,I))
    print('Theory: mean:%.3f Var:%.3f'%(mean,std**2))
    return p_value_norm(Cmin,mean,std)

def bootstrap_test(Cmin,beta,n,B=1000):
    mu_hat=beta[0]
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
    k=0
    for i in range(B):
        if C[i]>=Cmin:
            k+=1
    return k/B

def bootstrap_CAN(Cmin,beta,n,B=1000):
    mu_hat=beta[0]
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
    print('Bootstrap_CAN: mean:%.3f Var:%.3f'%(statistics.mean(C), statistics.variance(C)))
    return p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C))

#def double_boostrap(Cmin,beta,B1=1000,B2=1000):
#    mu_hat=beta[0]
#    C=[0 for x in range(B1)]
#    pvalue=[0 for x in range(B1)]
#    for i in range(B1):
#        print(i)
#        x = poisson_data_constant(mu_hat, n)
#        mu_mle = np.mean(x)
#        r = generate_s_constant(n, mu_mle)
#        for j in range(n):
#            if x[j] == 0:
#                C[i] += r[j]
#            else:
#                C[i] += r[j] - x[j] * math.log(r[j] / x[j], math.e) - x[j]
#        C[i] = 2 * C[i]  # 2 is here
#        pvalue[i]=bootstrap_test(C[i],[mu_mle],B2)
#    k = 0
#    for i in range(B1):
#        if C[i] >= Cmin:
#            k += 1
#    p=k/B1
#    l=0
#    for i in range(B1):
#        if pvalue[i]<=p:
#            l+=1
#    return l/B1

def double_boostrap(Cmin,beta,n,B1=1000,B2=1000):
    mu_hat=beta[0]
    mu_tilde=[0 for x in range(B1)]
    for i in range(B1):
        x = poisson_data_constant(mu_hat, n)
        mu_tilde[i]=np.mean(x)
    mu_adj=2*mu_hat-np.mean(mu_tilde)
    return bootstrap_test(Cmin,[mu_adj],n,B2)

path='countFormat1-segment-3-4.dat'
count=np.genfromtxt(path,skip_header=0)
print(path)
data=count[:,1]

n=len(data)
B=1000
B1=1000
B2=1000

max=30
X = np.mat([[1 for x in range(n)]]).T
I = np.mat([1.]).T

mu_hat=np.mean(data)

r = generate_s_constant(n, mu_hat)
Cmin = 0
for j in range(n):
    if data[j] == 0:
        Cmin += 2 * r[j]
    else:
        Cmin += 2 * (r[j] - data[j] * math.log(r[j], math.e) - data[j] + data[j] * math.log(data[j], math.e))
print('n: %d' %(n),'mu_hat: %.3f '%(mu_hat),'Cmin: %.3f' %(Cmin))

print('p-values:','Alg.1: %.3f\t'%(p_value_chi(Cmin,n-1)),'Alg.2: %.3f\t'%(bootstrap_CAN(Cmin,[mu_hat],n,B)),'Alg.3: %.3f\t'%(theory_test(Cmin,[mu_hat],n,X,I,max)),'Alg.4: %.3f\t'%(bootstrap_test(Cmin,[mu_hat],n,B)),'Alg.4b: %.3f'%(double_boostrap(Cmin,[mu_hat],n,B1,B2)))
