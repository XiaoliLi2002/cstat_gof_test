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
plt.figure(dpi=300,figsize=(18,12))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

def chi2fun(x,n):
    pdf=1/(2**(n/2)*scipy.special.gamma(n/2))*x**(n/2-1)*math.e**(-x/2)
    return pdf

def func(x,theta):  # Powerlaw model
    return theta[0]*x**(-theta[1])

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
    x=kapa12(s[i],max)/s[i]**2
    for j in range(n):
        x-=kapa11(s[j],max)*Q[j,i]/s[j]*kapa03(s[i])/s[i]**3
    return x

def expectation(s,n,X,I,max):
    V = np.diag([1/s[i] for i in range(n)])
    #print(V)
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n,max) for i in range(n)])
    print(-0.5 * (X.T * V * X) ** (-1)*X.T * Sigma * X)
    print(-0.5 * X.T * Sigma * X * (X.T * V * X) ** (-1))
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    print(E)
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

def theory_test(Cmin,beta,n,X,I,ARE,RMF,Energy,BACK=0.,left=0,right=0):
    s = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    m=(int)(max(s))+1
    if m<=20:
        maximum=40
    else:
        maximum=2*m
    mean=expectation(s,n,X,I,maximum)
    std=math.sqrt(Var(s,n,X,I,maximum))
    x=np.arange(mean-3*std,mean+3*std,0.05)
    y=normfun(x,mean,std)
    plt.plot(x, y, color='tomato', label='Alg.3', linestyle='dashdot')
    return p_value_norm(Cmin,mean,std)

def bootstrap_test(Cmin,beta,n,ARE,RMF,Energy,BACK=0.,left=0,right=0,B=1000):
    C = [0 for x in range(B)]
    s=RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
    for i in range(B):
        print(i)
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

    plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4')
    #plt.title(r'$n=10,\mu=0.5$', fontsize=18)
    plt.xlabel('C-stat', fontsize=18)
    plt.ylabel('Density', fontsize=18)

    print(np.mean(C),np.std(C))
    x=np.arange(np.mean(C)-3*np.std(C),np.mean(C)+3*np.std(C),0.05)
    y=normfun(x,np.mean(C),np.std(C))
    plt.plot(x, y, color='grey', label='Alg.2')

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
    print(statistics.mean(C), statistics.stdev(C))
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

def RMF_si_powerlaw(ARE,RMF,Energy,i,theta,BACK=0.,left=0,right=0): #len(Energy)=len(ARE)+1
    si=BACK
    for j in range(len(ARE)):
        si+=RMF[j,i+left]*ARE[j]*func((Energy[j]+Energy[j+1])/2,theta)*(Energy[j+1]-Energy[j])
    return si

def design_mat(ARE,RMF,Energy,theta,left=0,right=0):
    n=len(ARE)-left-right
    p=len(theta)
    X=np.mat([ [0. for l in range(p)] for k in range(n) ])
    for k in range(n):
        X[k,0]=RMF_si_powerlaw(ARE,RMF,Energy,k,theta,BACK=0,left=left,right=right)/theta[0]
        for l in range(len(ARE)):
            X[k,1]+=-theta[1]*RMF[l,k+left]*ARE[l]*func((Energy[l]+Energy[l+1])/2,theta)*np.log((Energy[l]+Energy[l+1])/2)*(Energy[l+1]-Energy[l])
    return X

def LLF(theta,x,ARE,RMF,Energy,BACK=0,left=0,right=0):
    n = len(x)
    s = RMF_s_powerlaw(ARE,RMF,Energy,theta,BACK,left,right)
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
    return -value

def LLF2(theta,x,ARE,RMF,Energy,BACK=0,left=0,right=0):
    n = len(x)
    I=np.eye(n)
    s = RMF_s_powerlaw(ARE,I,Energy,theta,BACK,left,right)
    value=0
    for i in range(n):
        value+=x[i]*math.log(s[i],math.e)-s[i]
    return -value

def tridiag_mat(n,shift,value):
    X=np.eye(n)-np.eye(n)
    for i in range(n-shift):
        X[i,i+shift]=value
    return X

def test(n,B,B1,B2,beta,iters,ARE,RMF,Energy,BACK=0.,left=0,right=0):

    for l in range(iters):
        plt.subplot(2, 1, 1)
        I = np.eye(n)
        C = [0 for x in range(B)]
        C2 = [0 for x in range(B)]
        s = RMF_s_powerlaw(ARE, RMF_test1, Energy, beta, BACK, left, right)
        s2 = RMF_s_powerlaw(ARE, I, Energy, beta, BACK, left, right)

        for i in range(B):
            print(i)
            x = poisson_data(s)
            if x == [0 for i in range(n)]:
                continue
            xopt = opt.minimize(LLF, beta, args=(x, ARE, RMF_test1, Energy, BACK, left, right),
                                bounds=([[1e-5, np.inf], [-10, 10]]))
            theta_hat = xopt['x']
            r = RMF_s_powerlaw(ARE, RMF_test1, Energy, theta_hat, BACK, left, right)
            for j in range(n):
                if x[j] == 0:
                    C[i] += r[j]
                else:
                    C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
            C[i] = 2 * C[i]  # 2 is here

        for i in range(B):
            print(i)
            x = poisson_data(s2)
            if x == [0 for i in range(n)]:
                continue
            xopt = opt.minimize(LLF2, beta, args=(x, ARE, I, Energy, BACK, left, right),
                                bounds=([[1e-5, np.inf], [-10, 10]]))
            theta_hat = xopt['x']
            r = RMF_s_powerlaw(ARE, I, Energy, theta_hat, BACK, left, right)
            for j in range(n):
                if x[j] == 0:
                    C2[i] += r[j]
                else:
                    C2[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
            C2[i] = 2 * C2[i]  # 2 is here

        plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4,RMF')
        plt.xlabel('C-stat', fontsize=18)
        plt.ylabel('Density', fontsize=18)

        x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
        y = normfun(x, statistics.mean(C), statistics.stdev(C))
        plt.plot(x, y, color='red', label='Alg.2b,RMF')

        plt.hist(C2, bins=20, rwidth=0.9, density=True, color='orange', label='Alg.4,noRMF')

        x = np.arange(statistics.mean(C2) - 3 * statistics.stdev(C2), statistics.mean(C2) + 3 * statistics.stdev(C2), 0.05)
        y = normfun(x, statistics.mean(C2), statistics.stdev(C2))
        plt.plot(x, y, color='green', label='Alg.2b,noRMF')

        plt.legend(fontsize=15)
        plt.grid(linestyle='--', alpha=0.5)
        # plt.show()
        plt.title(r'$n=100,\frac{K}{n}=1,\alpha=1,R=D$', fontsize=18)

        plt.subplot(2, 1, 2)
        I=np.eye(n)
        C = [0 for x in range(B)]
        C2 = [0 for x in range(B)]
        s = RMF_s_powerlaw(ARE, RMF_test2, Energy, beta, BACK, left, right)
        s2 = RMF_s_powerlaw(ARE, I, Energy, beta, BACK, left, right)

        for i in range(B):
            print(i)
            x = poisson_data(s)
            if x == [0 for i in range(n)]:
                continue
            xopt = opt.minimize(LLF, beta, args=(x, ARE, RMF_test2, Energy, BACK, left, right),
                                bounds=([[1e-5, np.inf], [-10, 10]]))
            theta_hat = xopt['x']
            r = RMF_s_powerlaw(ARE, RMF_test2, Energy, theta_hat, BACK, left, right)
            for j in range(n):
                if x[j] == 0:
                    C[i] += r[j]
                else:
                    C[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
            C[i] = 2 * C[i]  # 2 is here

        for i in range(B):
            print(i)
            x = poisson_data(s2)
            if x == [0 for i in range(n)]:
                continue
            xopt = opt.minimize(LLF2, beta, args=(x, ARE, I, Energy, BACK, left, right),
                                bounds=([[1e-5, np.inf], [-10, 10]]))
            theta_hat = xopt['x']
            r = RMF_s_powerlaw(ARE, I, Energy, theta_hat, BACK, left, right)
            for j in range(n):
                if x[j] == 0:
                    C2[i] += r[j]
                else:
                    C2[i] += r[j] - x[j] * np.log(r[j] / x[j]) - x[j]
            C2[i] = 2 * C2[i]  # 2 is here

        plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4,RMF')
        plt.xlabel('C-stat', fontsize=18)
        plt.ylabel('Density', fontsize=18)

        x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
        y = normfun(x, statistics.mean(C), statistics.stdev(C))
        plt.plot(x, y, color='red', label='Alg.2b,RMF')

        plt.hist(C2, bins=20, rwidth=0.9, density=True, color='orange', label='Alg.4,noRMF')

        x = np.arange(statistics.mean(C2) - 3 * statistics.stdev(C2), statistics.mean(C2) + 3 * statistics.stdev(C2),
                      0.05)
        y = normfun(x, statistics.mean(C2), statistics.stdev(C2))
        plt.plot(x, y, color='green', label='Alg.2b,noRMF')

        plt.legend(fontsize=15)
        plt.grid(linestyle='--', alpha=0.5)
        # plt.show()
        plt.title(r'$n=100,\frac{K}{n}=1,\alpha=1,R=\frac{(11^\top+nI)}{2n}$', fontsize=18)
        plt.savefig('RMFvsnoRMF.pdf')

B=1000
n_test=100
ARE_test=np.array([1 for i in range(n_test)])
#RMF_test1=np.eye(n_test)
RMF_test1=0.8*np.eye(n_test)+tridiag_mat(n_test,1,0.1)+tridiag_mat(n_test,1,0.1).T
RMF_test1[0,0]+=0.1
RMF_test1[n_test-1,n_test-1]+=0.1
RMF_test2=np.mat([ [.5 for l in range(n_test) ] for k in range(n_test)])/n_test+0.5*np.eye(n_test)
Energy_test=np.array([1.+i/n_test for i in range(n_test+1)])
beta_test=np.array([1*n_test,1])
#beta_test=np.array([2.,1])
iters_test=1

test(n_test,B,B,B,beta_test,iters_test,ARE_test,RMF_test2,Energy_test,BACK=0.1,left=0,right=0)