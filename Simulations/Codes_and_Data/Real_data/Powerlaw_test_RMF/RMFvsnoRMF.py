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

def func(x,theta):  # Powerlaw model
    return theta[0]*x**(-theta[1])

def RMF_si_powerlaw(ARE,RMF,Energy,i,theta,BACK=0.,left=0,right=0): #len(Energy)=len(ARE)+1
    si=BACK
    for j in range(len(ARE)):
        si+=RMF[j,i+left]*ARE[j]*func((Energy[j]+Energy[j+1])/2,theta)*(Energy[j+1]-Energy[j])
    return si

def RMF_s_powerlaw(ARE,RMF,Energy,theta,BACK=0.,left=0,right=0):
    n=len(ARE)-left-right
    s=[0 for i in range(n)]
    for i in range(n):
        s[i]=RMF_si_powerlaw(ARE,RMF,Energy,i,theta,BACK,left,right)
    return s

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def poisson_data(mu):
    arr=[np.random.poisson(x) for x in mu]
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
    x=kapa12(s[i],max)/s[i]**2
    for j in range(n):
        x-=kapa11(s[j],max)*Q[j,i]/s[j]*kapa03(s[i])/s[i]**3
    return x

def expectation(s,n,X,I,max):
    V = np.diag([1/s[i] for i in range(n)])
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n,max) for i in range(n)])
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    for i in range(n):
        E += kapa1(s[i],max)
    print(E)
    return float(E)

def Var(s,n,X,I,max):
    p=len(I)
    V = np.diag([1/s[i] for i in range(n)])
    k_11 = np.mat([kapa11(s[i],max)/s[i] for i in range(n)]).T
    var = (-k_11.T * X * (X.T * V * X) ** (-1) * X.T * k_11)[0, 0]
    for i in range(n):
        var += kapa2(s[i],max)
    print(math.sqrt((1-p/n)*var))
    return (1-p/n)*var

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

def theory_test(x,beta,n,X,I,ARE,RMF,Energy,BACK=0.,left=0,right=0):
    s = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    m = (int)(max(s)) + 1
    if m <= 20:
        maximum = 40
    else:
        maximum = 2 * m
    return p_value_norm(x,expectation(s,n,X,I,maximum),math.sqrt(Var(s,n,X,I,maximum)))

def bootstrap_test(Cmin,beta,n,ARE,RMF,Energy,BACK=0.,left=0,right=0,B=1000):
    C = [0 for x in range(B)]
    s=RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    for i in range(B):
        x = poisson_data(s)
        if x==[0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta_hat = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
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

def bootstrap_CAN(Cmin,beta,n,ARE,RMF,Energy,BACK=0.,left=0,right=0,B=1000):
    C = [0 for x in range(B)]
    s = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta_hat = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
        for j in range(n):
            if x[j] == 0:
                C[i] += r[j]
            else:
                C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
        C[i] = 2 * C[i]  # 2 is here
    print(np.mean(C), np.std(C))
    return p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C))

def double_boostrap(Cmin,beta,n,ARE,RMF,Energy,BACK=0.,left=0,right=0,B1=1000,B2=1000):
    C=[0 for x in range(B1)]
    pvalue=[0 for x in range(B1)]
    for i in range(B1):
        s=RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
        x=poisson_data(s)
        if x == [0 for j in range(n)]:
            pvalue[i]=1.
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta_hat = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
        for j in range(n):
            if x[j] == 0:
                C[i] += r[j]
            else:
                C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
        C[i] = 2 * C[i]  # 2 is here
        pvalue[i]=bootstrap_test(C[i],theta_hat,n,ARE,RMF,Energy,BACK,left,right,B2)
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

def bootstrap_bias(Cmin,beta,n,ARE,RMF,Energy,BACK=0.,left=0,right=0,B1=1000,B2=1000):
    theta1_hat=beta[0]
    theta2_hat=beta[1]
    theta1_tilde=[0 for x in range(B1)]
    theta2_tilde=[0 for x in range(B1)]
    s=RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    for i in range(B1):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, [beta[0], beta[1]], args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta1_tilde[i] = xopt['x'][0]
        theta2_tilde[i] = xopt['x'][1]
    theta1_adj=2*theta1_hat-np.mean(theta1_tilde)
    if theta1_adj<=1e-7:
        theta1_adj=0
    theta2_adj = 2 * theta2_hat - np.mean(theta2_tilde)
    return bootstrap_test(Cmin,[theta1_adj,theta2_adj],n,B2)

def design_mat(ARE,RMF,Energy,theta,left=0,right=0):
    n=len(ARE)-left-right
    p=len(theta)
    X=np.mat([ [0. for l in range(p)] for k in range(n) ])
    for k in range(n):
        X[k,0]=RMF_si_powerlaw(ARE,RMF,Energy,k,theta,BACK=0,left=left,right=right)/theta[0]
        for l in range(len(ARE)):
            X[k,1]+=-theta[1]*RMF[l,k+left]*ARE[l]*func((Energy[l]+Energy[l+1])/2,theta)*np.log((Energy[l]+Energy[l+1])/2)*(Energy[l+1]-Energy[l])
    return X

def test(n,B,B1,B2,beta,iters,ARE,RMF,Energy,BACK=0.,left=0,right=0):
    p=len(beta)
    I=np.mat([1. for i in range(p)]).T

    reject1 = [0 for y in range(iters)]  # rmf chi2
    reject2 = [0 for y in range(iters)]  # rmf theory
    reject3 = [0 for y in range(iters)]  # nonrmf chi2
    reject4 = [0 for y in range(iters)]  # nonrmf theory


    for l in range(iters):
        print(l)
        s = RMF_s_powerlaw(ARE,RMF,Energy,beta,BACK,left,right)
        x = poisson_data(s)  #true is with RMF

        #RMF
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, np.array([beta[0], beta[1]]),args=(x,ARE,RMF,Energy,BACK,left,right),bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta_hat=xopt['x']
        print(theta_hat)
        r = RMF_s_powerlaw(ARE,RMF,Energy,theta_hat,BACK,left,right)
        Cmin = 0.
        for j in range(n):
            if x[j] == 0:
                Cmin += r[j]
            else:
                Cmin += r[j] - x[j] * math.log(r[j] / x[j], math.e) - x[j]
        Cmin = 2 * Cmin
        print(Cmin)
        if p_value_chi(Cmin, n - 2) < 0.05:
            reject1[l] = 1
        X=design_mat(ARE,RMF,Energy,theta_hat,left,right)
        if theory_test(Cmin, theta_hat,n,X,I,ARE,RMF,Energy,BACK,left,right) < 0.05:
            reject2[l] = 1

        #nonRMF
        eye=np.eye(n)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, np.array([beta[0], beta[1]]),args=(x,ARE,eye,Energy,BACK,left,right),bounds=([[1e-5, 30*n_test], [-50, 10]]))
        theta_hat=xopt['x']
        print(theta_hat)
        r = RMF_s_powerlaw(ARE,eye,Energy,theta_hat,BACK,left,right)
        Cmin = 0.
        for j in range(n):
            if x[j] == 0:
                Cmin += r[j]
            else:
                Cmin += r[j] - x[j] * math.log(r[j] / x[j], math.e) - x[j]
        Cmin = 2 * Cmin
        print(Cmin)
        if p_value_chi(Cmin, n - 2) < 0.05:
            reject3[l] = 1
        X=design_mat(ARE,eye,Energy,theta_hat,left,right)
        if theory_test(Cmin, theta_hat,n,X,I,ARE,eye,Energy,BACK,left,right) < 0.05:
            reject4[l] = 1

    print(np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4))
    result=[beta[0],np.mean(reject1), np.mean(reject2), np.mean(reject3), np.mean(reject4)]
    file1=open('result_rmfvsnormf.txt','a')
    print(result,file=file1)
    file1.close()

def LLF(theta,x,ARE,RMF,Energy,BACK=0,left=0,right=0):
    n = len(x)
    s = RMF_s_powerlaw(ARE,RMF,Energy,theta,BACK,left,right)
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
    return -value

def tridiag_mat(n,shift,value):
    X=np.eye(n)-np.eye(n)
    for i in range(n-shift):
        X[i,i+shift]=value
    return X

def LLF2(theta,x,ARE,RMF,Energy,BACK=0,left=0,right=0):
    n = len(x)
    I=np.eye(n)
    s = RMF_s_powerlaw(ARE,I,Energy,theta,BACK,left,right)
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
    return -value

B=100
n_test=100
ARE_test=np.array([1 for i in range(n_test)])
#RMF_test=np.diag([1 for j in range(n_test)])
RMF_test=0.8*np.eye(n_test)+tridiag_mat(n_test,1,0.1)+tridiag_mat(n_test,1,0.1).T
RMF_test[0,0]+=0.1
RMF_test[n_test-1,n_test-1]+=0.1
RMF_test2=np.mat([ [.5 for l in range(n_test) ] for k in range(n_test)])/n_test+0.5*np.eye(n_test)
Energy_test=np.array([1.+i/n_test for i in range(n_test+1)])
beta_test=np.array([0.5*n_test,1.])
iters_test=1000

test(n_test,B,B,B,beta_test,iters_test,ARE_test,RMF_test2,Energy_test,BACK=0.1,left=0,right=0)