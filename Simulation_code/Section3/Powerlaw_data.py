import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import nbinom
import scipy.optimize as opt
import scipy
import time
np.set_printoptions(threshold=np.inf)

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=np.random.poisson(mu[i])
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
        arr[i]=theta1*((1+(i+1)/n)**(-theta2))
    return arr

def generate_s_norm(n,mu,sigma):
    return np.random.normal(mu,sigma,n)

def generate_s_gamma(n,alpha,beta):
    return scipy.stats.gamma.rvs(alpha,loc=0,scale=1/beta,size=n)

def poisson_dis(mu,i):
    return mu**i/math.factorial(i)*math.e**(-mu)

def Sigma_diag(s,i,Q,n,max):
    x=kapa12(s[i],max)
    for j in range(n):
        x-=kapa11(s[j],max)*Q[j,i]*kapa03(s[i])
    return x

def expectation(beta,n,X,I,max):
    s = generate_s_powerlaw(n, beta[0], beta[1])
    V = np.diag([s[i] for i in range(n)])
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n,max) for i in range(n)])
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    for i in range(n):
        E += kapa1(s[i],max)
    return float(E)

def Var(beta,n,X,I,max):
    s = generate_s_powerlaw(n, beta[0], beta[1])
    V = np.diag([s[i] for i in range(n)])
    k_11 = np.mat([kapa11(s[i],max) for i in range(n)]).T
    var = (-k_11.T * X * (X.T * V * X) ** (-1) * X.T * k_11)[0, 0]
    for i in range(n):
        var += kapa2(s[i],max)
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

def theory_test(x,beta,n,X,I,max=30):
    return p_value_norm(x,expectation(beta,n,X,I,max),math.sqrt(Var(beta,n,X,I,max)))

def bootstrap_test(Cmin,beta,n,B=1000):
    C = [0 for x in range(B)]
    s=generate_s_powerlaw(n,beta[0],beta[1])
    for i in range(B):
        x = poisson_data(s)
        if x==[0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]],x , bounds=([[1e-5, 30], [-50, 10]]))
        theta1_hat = xopt['x'][0]
        theta2_hat = xopt['x'][1]
        r = generate_s_powerlaw(n, theta1_hat, theta2_hat)
        for j in range(n):
            if x[j] == 0:
                C[i] +=r[j]
            else:
                C[i] +=r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
        C[i]=2*C[i] # 2 is here
    k=0
    for i in range(B):
        if C[i]>=Cmin:
            k+=1
    return k/B

def bootstrap_CAN(Cmin,beta,n,B=1000):
    C = [0 for x in range(B)]
    s = generate_s_powerlaw(n, beta[0], beta[1])
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], x, bounds=([[1e-5, 30], [-50, 10]]))
        theta1_hat = xopt['x'][0]
        theta2_hat = xopt['x'][1]
        r = generate_s_powerlaw(n, theta1_hat, theta2_hat)
        for j in range(n):
            if x[j] == 0:
                C[i] += r[j]
            else:
                C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
        C[i] = 2 * C[i]  # 2 is here
    return p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C))

def double_boostrap(Cmin,beta,n,B1=1000,B2=1000):
    C=[0 for x in range(B1)]
    pvalue=[0 for x in range(B1)]
    for i in range(B1):
        s=generate_s_powerlaw(n,beta[0],beta[1])
        x=poisson_data(s)
        if x == [0 for j in range(n)]:
            pvalue[i]=1.
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], x, bounds=([[1e-5, 30], [-50, 10]]))
        theta1_hat = xopt['x'][0]
        theta2_hat = xopt['x'][1]
        r = generate_s_powerlaw(n, theta1_hat, theta2_hat)
        for j in range(n):
            if x[j] == 0:
                C[i] += r[j]
            else:
                C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
        C[i] = 2 * C[i]  # 2 is here
        pvalue[i]=bootstrap_test(C[i],[theta1_hat,theta2_hat],n,B2)
    k = 0
    for i in range(B1):
        if C[i] >= Cmin:
            k += 1
    p=k/B1
    l=0
    for i in range(B1):
        if pvalue[i]<=p:
            l+=1
    return l/B1

def bootstrap_bias(Cmin,beta,n,B1=1000,B2=1000):
    theta1_hat=beta[0]
    theta2_hat=beta[1]
    theta1_tilde=[0 for x in range(B1)]
    theta2_tilde=[0 for x in range(B1)]
    s=generate_s_powerlaw(n,theta1_hat,theta2_hat)
    for i in range(B1):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], x, bounds=([[1e-5, 30], [-50, 10]]))
        theta1_tilde[i] = xopt['x'][0]
        theta2_tilde[i] = xopt['x'][1]
    theta1_adj=2*theta1_hat-np.mean(theta1_tilde)
    if theta1_adj<=1e-7:
        theta1_adj=0
    theta2_adj = 2 * theta2_hat - np.mean(theta2_tilde)
    return bootstrap_test(Cmin,[theta1_adj,theta2_adj],n,B2)

def test(n,B,B1,B2,beta,max,X,I,iters,k):
    reject1 = [0 for y in range(iters)]  # chi2
    reject2 = [0 for y in range(iters)]  # CANB
    reject3 = [0 for y in range(iters)]  # Highorder
    reject4 = [0 for y in range(iters)]  # B
    reject5 = [0 for y in range(iters)]  # B.b.c
    reject6 = [0 for y in range(iters)]  # double B

    for l in range(iters):
        print(k,l)
        # s = generate_s_exp(n,5,3)
        s = generate_s_powerlaw(n,beta[0],beta[1])
        print(s)
        # s=generate_s_constant(n,0.01)
        # s = generate_s_gamma(n,0.25,math.sqrt(0.25))

        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], x, bounds=([[1e-5, 30], [-50, 10]]))
        theta1_hat = xopt['x'][0]
        theta2_hat = xopt['x'][1]
        r = generate_s_powerlaw(n, theta1_hat, theta2_hat)
        Cmin = 0.
        for j in range(n):
            if x[j] == 0:
                Cmin += r[j]
            else:
                Cmin += r[j] - x[j] * math.log(r[j] / x[j], math.e) - x[j]
        Cmin = 2 * Cmin

        if p_value_chi(Cmin, n - 2) < 0.05:
            reject1[l] = 1
        start = time.time()
        if bootstrap_CAN(Cmin, [theta1_hat, theta2_hat], n, B) < 0.05:
            reject2[l] = 1
        end = time.time()
        print("alg.2", end - start)
        if theory_test(Cmin, [theta1_hat, theta2_hat], n, X, I, max) < 0.05:
            reject3[l] = 1
        start = time.time()
        print('alg.3', start - end)
        if bootstrap_test(Cmin, [theta1_hat, theta2_hat], n, B) < 0.05:
            reject4[l] = 1
        end = time.time()
        print('alg.4', end - start)
        if bootstrap_bias(Cmin, [theta1_hat, theta2_hat], n, B1, B2) < 0.05:
            reject5[l] = 1
        start = time.time()
        print('bias', start - end)
        if double_boostrap(Cmin, [theta1_hat, theta2_hat], n, B1, B2) < 0.05:
            reject6[l] = 1
        end = time.time()
        print('double b', end - start)
    print(np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4), np.mean(reject5), np.mean(reject6))
    error1[k] = np.mean(reject1)
    error2[k] = np.mean(reject2)
    error3[k] = np.mean(reject3)
    error4[k] = np.mean(reject4)
    error5[k] = np.mean(reject5)
    error6[k] = np.mean(reject6)
    result=[np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4), np.mean(reject5), np.mean(reject6)]
    file1=open('result.txt','a')
    print(result,file=file1)
    file1.close()

def LLF(theta,x):
    n = len(x)
    s = generate_s_powerlaw(n,theta[0],theta[1])
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
    return -value

n = 30
B = 100
B1 = 100
B2 = 100
theta1 = .5
theta2 = 1
max = 30
X = np.mat([[1 for x in range(n)], [math.log(1+(x + 1) / n, math.e) for x in range(n)]]).T
I = np.mat([1., 1.]).T
iters = 100

repetition=20
error1 = [0 for y in range(repetition)]  # chi2
error2 = [0 for y in range(repetition)]  # CANB
error3 = [0 for y in range(repetition)]  # Highorder
error4 = [0 for y in range(repetition)]  # B
error5 = [0 for y in range(repetition)]  # B_bc
error6 = [0 for y in range(repetition)]  # pDB

for k in range(repetition):
    iters=100
    test(n,B,B1,B2,[theta1, theta2],max,X,I,iters,k)
print('Finished!')
print(np.mean(error1),np.mean(error2),np.mean(error3),np.mean(error4),np.mean(error5),np.mean(error6))
print(np.std(error1),np.std(error2),np.std(error3),np.std(error4),np.std(error5),np.std(error6))

results=[[np.mean(error1),np.mean(error2),np.mean(error3),np.mean(error4),np.mean(error5),np.mean(error6)],
         [np.std(error1),np.std(error2),np.std(error3),np.std(error4),np.std(error5),np.std(error6)]]
file=open('results.txt','w')
print(results,file=file)
file.close()