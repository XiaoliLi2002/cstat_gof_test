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

def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

def LLF_exp(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*(math.log(theta[0],math.e)-theta[1]*(i+1)/n)-theta[0]*math.e**(-theta[1]*(i+1)/n)
    return -value

def LLF_powerlaw(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*math.log(theta[0]*((1+(i+1)/n)**theta[1]),math.e)-theta[0]*((1+(i+1)/n)**theta[1])
    return -value

def LLF_constant(mu):
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

max=300
def poisson_dis(mu,i):
    return poisson.pmf(i,mu)

def Sigma_diag(s,i,Q,n):
    x=kapa_12
    for j in range(n):
        x-=kapa_11*Q[j,i]*kapa_03
    return x

def expectation(beta,n):
    s=generate_s_constant(n,beta[0])
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])

    E=(-0.5*I.T*X.T*Sigma*X*(X.T*V*X)**(-1)*I)[0]
    for i in range(n):
        E+=kapa_1
    print(float(E))
    return float(E)

def Var(beta,n):
    s = generate_s_constant(n, beta[0])
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11 for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0]
    for i in range(n):
        var+=kapa_2
    print(math.sqrt(var))
    return var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))

blockLength=161
nBlocks=2
path="./count_data.xlsx"

B=1000
data=pd.DataFrame(pd.read_excel(path,sheet_name="Sheet1")).T
data=np.array(data)
print(data)
for i in range(nBlocks):
    x=data[i]
    print(x)
    #constant model
    print("Constant")
    X = np.mat([[1 for x in range(blockLength)]]).T
    I = np.mat([1.]).T
    mu_hat=np.mean(x)
    print(mu_hat)
    r = generate_s_constant(blockLength, mu_hat)
    Cmin = 0
    for j in range(blockLength):
        if x[j] == 0:
            Cmin += 2 * r[j]
        else:
            Cmin += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
    print(Cmin)
    s = generate_s_constant(blockLength, mu_hat)
    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        mu_mle=np.mean(x)
        r = generate_s_constant(blockLength, mu_hat)
        for j in range(blockLength):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
    print(statistics.mean(C), statistics.stdev(C), p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C)))
    tem = 0
    C_re = sorted(C)
    for i in range(B):
        if C_re[i] < Cmin:
            i += 1
            tem += 1
        else:
            print((B - i) / B)
            break
    print(p_value_chi(Cmin, blockLength - 1))

    x = 0
    for k in range(max):
        if k == 0:
            x -= 2 * (k - mu_hat) * poisson_dis(mu_hat, k)
        else:
            x += 2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) * poisson_dis(mu_hat, k)
    kapa_1 = x

    k_1 = kapa_1
    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) ** 2 * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) ** 2 * poisson_dis(mu_hat, k)
    kapa_2 = x

    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) * (k - mu_hat) * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) * (k - mu_hat) * poisson_dis(mu_hat, k)
    kapa_11 = x

    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu_hat) - k_1) * (k - mu_hat) ** 2 * poisson_dis(mu_hat, k)
        else:
            x += (2 * (k * math.log(k / mu_hat, math.e) - k + mu_hat) - k_1) * (k - mu_hat) ** 2 * poisson_dis(mu_hat,
                                                                                                               k)
    kapa_12 = x

    kapa_03 = mu_hat

    print(theory_test(Cmin, [mu_hat], blockLength))

