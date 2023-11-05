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
from sherpa.astro import io
from sherpa.astro import instrument
from sherpa.astro.plot import ARFPlot
from sherpa.utils.testing import get_datadir
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
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n,max) for i in range(n)])
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    for i in range(n):
        E += kapa1(s[i],max)
    print(E)
    return float(E)

def Var(s,n,X,I,max):
    V = np.diag([1/s[i] for i in range(n)])
    k_11 = np.mat([kapa11(s[i],max)/s[i] for i in range(n)]).T
    var = (-k_11.T * X * (X.T * V * X) ** (-1) * X.T * k_11)[0, 0]
    for i in range(n):
        var += kapa2(s[i],max)
    print(math.sqrt(var))
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

def Cashstat(x,s):
    C=0.
    for j in range(len(x)):
        if x[j] <1e-5:
            C+=s[j]
        else:
            C +=s[j] - x[j] * np.log(s[j] / x[j]) - x[j]
    C=C*2
    return C

def theory_test(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0):
    n=len(counts)
    if (counts==np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    r = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat=Cashstat(counts,r)
    X=design_mat(ARE,RMF,Energy,theta_hat,left,right)
    p=len(theta_hat)
    I=np.mat([1. for i in range(p)]).T
    m=(int)(max(r))+1
    if m<=20:
        maximum=40
    else:
        maximum=2*m
    return p_value_norm(Cstat,expectation(r,n,X,I,maximum),math.sqrt(Var(r,n,X,I,maximum)))

def bootstrap_test(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0,B=1000):
    n = len(counts)
    if (counts==np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    s = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat = Cashstat(counts, s)
    C = np.zeros(B)
    for i in range(B):
        begin=time.time()
        x = poisson_data(s)
        if x==[0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, theta_hat, args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-8, np.inf], [-10, 10]]))
        beta = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
        C[i]=Cashstat(x,r)
        last=time.time()
        print("iter",i,"using",last-begin,"s")
    print("Bootstrap using asymptotic normality:",p_value_norm(Cstat, np.mean(C), np.std(C)))
    k=0
    for i in range(B):
        if C[i]>=Cstat:
            k+=1
    return k/B

def bootstrap_CAN(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0,B=1000):
    n = len(counts)
    if (counts==np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    s = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat = Cashstat(counts, s)
    C = np.zeros(B)
    for i in range(B):
        print(i)
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, theta_hat, args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-8, np.inf], [-10, 10]]))
        beta = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
        C[i] = Cashstat(x, r)
    print(np.mean(C), np.std(C))
    return p_value_norm(Cstat, statistics.mean(C), statistics.stdev(C))

def double_boostrap(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0,B1=1000,B2=1000):
    n = len(counts)
    if (counts==np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    s = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat = Cashstat(counts, s)
    C=np.zeros(B1)
    pvalue=np.zeros(B1)
    for i in range(B1):
        print("double",i)
        x=poisson_data(s)
        if x == [0 for j in range(n)]:
            pvalue[i]=1.
            continue
        xopt = opt.minimize(LLF, theta_hat, args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-8, np.inf], [-10, 10]]))
        beta = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
        C[i] = Cashstat(x, r)
        pvalue[i]=bootstrap_test(x,ARE,RMF,Energy,BACK,left,right,B2)
    k = 0
    for i in range(B1):
        if C[i] >= Cstat:
            k += 1
    p=k/B1
    l=0
    for i in range(B1):
        if pvalue[i]<=p:
            l+=1
    return l/B1

def bootstrap_bias(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0,B1=1000,B2=1000):
    n = len(counts)
    if (counts==np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    theta1_hat=theta_hat[0]
    theta2_hat=theta_hat[1]
    theta1_tilde=np.zeros(B1)
    theta2_tilde=np.zeros(B1)
    s=RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat = Cashstat(counts, s)
    for i in range(B1):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, theta_hat, args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-8, np.inf], [-10, 10]]))
        theta1_tilde[i] = xopt['x'][0]
        theta2_tilde[i] = xopt['x'][1]
    theta1_adj=2*theta1_hat-np.mean(theta1_tilde)
    if theta1_adj<=1e-7:
        theta1_adj=0
    theta2_adj = 2 * theta2_hat - np.mean(theta2_tilde)
    s = RMF_s_powerlaw(ARE, RMF, Energy, [theta1_adj,theta2_adj], BACK, left, right)
    C = np.zeros(B)
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for i in range(n)]:
            continue
        xopt = opt.minimize(LLF, theta_hat, args=(x, ARE, RMF, Energy, BACK, left, right),
                            bounds=([[1e-8, np.inf], [-10, 10]]))
        beta = xopt['x']
        r = RMF_s_powerlaw(ARE, RMF, Energy, beta, BACK, left, right)
        C[i] = Cashstat(x, r)
    k = 0
    for i in range(B):
        if C[i] >= Cstat:
            k += 1
    return k / B

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

def tridiag_mat(n,shift,value):
    X=np.eye(n)-np.eye(n)
    for i in range(n-shift):
        X[i,i+shift]=value
    return X

def chi2test(counts,ARE,RMF,Energy,BACK=0.,left=0,right=0):
    n = len(counts)
    if (counts == np.zeros(n)).all():
        return 1
    xopt = opt.minimize(LLF, [1, 0], args=(counts, ARE, RMF, Energy, BACK, left, right),
                        bounds=([[1e-8, np.inf], [-10, 10]]))
    theta_hat = xopt['x']
    p=len(theta_hat)
    s = RMF_s_powerlaw(ARE, RMF, Energy, theta_hat, BACK, left, right)
    Cstat = Cashstat(counts, s)
    print(Cstat)
    return p_value_chi(Cstat,n-p)


B=1000
start=14705
end=14864
left=50
right=50

energy=io.read_arf("pg1116-oldobs_new_leg_abs1.arf").energ_lo[start-left:end+right+1]

oldBACK=0.002161
oldarf = io.read_arf("pg1116-oldobs_new_leg_abs1.arf").specresp[start-left:end+right]
oldrmf = io.read_rmf("pg1116-oldobs_new_leg_abs1.rmf")
oldrmfmatinfo=instrument.rmf_to_matrix(oldrmf)
oldmat=oldrmfmatinfo.matrix[start-left:end+right,start-left:end+right]
oldspec=io.read_pha("pg1116-oldobs_new_leg_abs1.pha").counts[start:end]
#print(len(energy),len(oldarf),oldmat.shape,len(oldspec))
pchi2=chi2test(oldspec,oldarf,oldmat,energy,oldBACK,left,right)
print("chi2:",pchi2)
ptheory=theory_test(oldspec,oldarf,oldmat,energy,oldBACK,left,right)
print("theory:",ptheory)
print("bootcan:",bootstrap_CAN(oldspec,oldarf,oldmat,energy,oldBACK,left,right,B))
pboot=bootstrap_test(oldspec,oldarf,oldmat,energy,oldBACK,left,right,B)



newBACK=0.00683
newarf = io.read_arf("pg1116-newobs_new_leg_abs1.arf").specresp[start-left:end+right]
newrmf = io.read_rmf("pg1116-newobs_new_leg_abs1.rmf")
newrmfmatinfo=instrument.rmf_to_matrix(newrmf)
newmat=newrmfmatinfo.matrix[start-left:end+right,start-left:end+right]
newspec=io.read_pha("pg1116-newobs_new_leg_abs1.pha").counts[start:end]
npchi2=chi2test(newspec,newarf,newmat,energy,newBACK,left,right)
print("chi2:",npchi2)
nptheory=theory_test(newspec,newarf,newmat,energy,newBACK,left,right,max)
print("theory:",nptheory)
npboot=bootstrap_test(newspec,newarf,newmat,energy,newBACK,left,right,B)
print("boot:",npboot)


print("old:",pchi2,ptheory,pboot,"new:",npchi2,nptheory,npboot)