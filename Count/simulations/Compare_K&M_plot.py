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
plt.figure(dpi=300,figsize=(18,12))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
#plt.rcParams['axes.linewidth'] = 2.0

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def chi2fun(x,n):
    pdf=1/(2**(n/2)*scipy.special.gamma(n/2))*x**(n/2-1)*math.e**(-x/2)
    return pdf

def generate_s(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*((1+(i+1)/n)**-theta2)
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
    value=0
    for i in range(n):
        value+=x[i]*math.log(theta[0]*((1+(i+1)/n)**-theta[1]),math.e)-theta[0]*((1+(i+1)/n)**-theta[1])
    return -value

def p_value_norm(x,mu,sigma):
    z=(x-mu)/sigma
    return scipy.stats.norm.sf(z)

def p_value_chi(x,df):
    return scipy.stats.chi2.sf(x,df)

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
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])
    print(-0.5*X.T * Sigma * X * (X.T * V * X) ** (-1))
    E=-0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    print(E)
    for i in range(n):
        E+=kapa_1(s[i])
    print(E)
    return float(E)

def expectation2(beta,n):
    s=generate_s(n,beta[0],beta[1])
    p=len(beta)
    E=0
    for i in range(n):
        E+=kapa_1(s[i])
    print((1-p/n)*E)
    return float((1-p/n)*E)

def Var(beta,n):
    s = generate_s(n, beta[0], beta[1])
    p=len(beta)
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0,0]
    for i in range(n):
        var+=kapa_2(s[i])
    print(math.sqrt((1-p/n)*var))
    return (1-p/n)*var

def theory_test(x,beta,n):
    return p_value_norm(x,expectation(beta,n),math.sqrt(Var(beta,n)))

def Kaastra_mean_unit(mu):
    if mu<=0.5:
        return -0.25*mu**3+1.38*mu**2-2*mu*math.log(mu,math.e)
    if mu>0.5 and mu<=2:
        return -0.00335*mu**5+0.04259*mu**4-0.27331*mu**3+1.381*mu**2-2*mu*math.log(mu,math.e)
    if mu>2 and mu<=5:
        return 1.019275+0.1345*mu**(0.461-0.9*math.log(mu,math.e))
    if mu>5 and mu<=10:
        return 1.00624+0.604*mu**(-1.68)
    if mu>10:
        return 1+0.1649/mu+0.226/mu**2

def Kaastra_mean(s):
    mean=0
    for x in s:
        mean+=Kaastra_mean_unit(x)
    return mean

def Kaastra_var_unit(mu):
    if mu>0.2 and mu<=0.3:
        return 4.23*mu**2-2.8254*mu+1.12522
    if mu>0.3 and mu<=0.5:
        return -3.7*mu**3+7.328*mu**2-3.6926*mu+1.20641
    if mu>0.5 and mu<=1:
        return 1.28*mu**4-5.191*mu**3+7.666*mu**2-3.5446*mu+1.15431
    if mu>1 and mu<=2:
        return 0.1125*mu**4-0.641*mu**3+0.859*mu**2+1.0914*mu-0.05748
    if mu>2 and mu<=3:
        return 0.089*mu**3-0.872*mu**2+2.8422*mu-0.67539
    if mu>3 and mu<=5:
        return 2.12336+0.012202*mu**(5.717-2.6*math.log(mu,math.e))
    if mu>5 and mu<=10:
        return 2.05159+0.331*mu**(1.343-math.log(mu,math.e))
    if mu>10:
        return 12/mu**3+0.79/mu**2+0.6747/mu+2

def Kaastra_var(s):
    var=0
    for x in s:
        var+=Kaastra_var_unit(x)
    return var

def Max_mean_unit(mu):
    A=-0.56709
    B= -2.7336
    C= -2.3603
    D= 0.52816
    E= 0.33133
    F=1.0174
    alpha=3.9375
    beta= 0.48446
    return (A+B*mu+C*(mu-D)**2)*math.e**(-alpha*mu)+E*math.e**(-beta*mu)+F

def Max_mean(s):
    mean=0
    for x in s:
        mean+=Max_mean_unit(x)
    return mean

def Max_var_unit(mu):
    A=-3.1971
    B= 1.5118
    C= -1.5118
    D= 0.79384
    E= 1.9294
    F=6.1740
    G= 22.360/1000
    H= -7.2981
    I= 2.08378
    alpha= 0.750315
    beta=4.49654
    return (A+B*mu**2+C*(mu-D)**2)*math.e**(-alpha*mu)+(E+F*mu+G*(mu-H)**2)*math.e**(-beta*mu)+I

def Max_var(s):
    var=0
    for x in s:
        var+=Max_var_unit(x)
    return var


B=2000

iters=1
for l in range(iters):
    plt.subplot(2, 2, 1)
    n=10
    X = np.mat([[1 for x in range(n)], [math.log(1 + (x + 1) / n, math.e) for x in range(n)]]).T
    I = np.mat([1., 1.]).T
    theta1 = 1.
    theta2 = 1.

    s = generate_s(n, theta1, theta2)
    #print(s)
    expec1 = expectation([theta1, theta2], n)
    expec2 = expectation2([theta1, theta2], n)
    std = math.sqrt(Var([theta1, theta2], n))

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        if x==[0 for y in range(n)]:
            continue
        xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, np.inf], [-30, 30]]))
        theta1_mle = xopt['x'][0]
        theta2_mle = xopt['x'][1]
        r = generate_s(n, theta1_mle, theta2_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    print(Kaastra_mean(s),Kaastra_var(s))
    #print(Max_mean(s),Max_var(s))

    #x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    #y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    #plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    #x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    #y = chi2fun(x, n - 2)
    #plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(0, 20, 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(Kaastra_mean(s) - 3 * math.sqrt(Kaastra_var(s)), Kaastra_mean(s) + 3 * math.sqrt(Kaastra_var(s)),
                  0.05)
    y = normfun(x, Kaastra_mean(s), math.sqrt(Kaastra_var(s)))
    plt.plot(x, y, color='blueviolet', label='Alg.2a')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color='green',linestyle='dashdot', label='Alg.2b')

    x = np.arange(expec2 - 3 * std, expec2 + 3 * std, 0.05)
    y = normfun(x, expec2, std)
    plt.plot(x, y, color='cyan', label='Alg.3a', marker='^',markevery=100)

    x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    y = normfun(x, expec1, std)
    plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True,color='cornflowerblue',label='Alg.4a')
    plt.xlabel('C-stat',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.xticks([0,5,10,15,20,25],fontsize=18)
    #plt.xticks([60,80,100,120,140,160],fontsize=18)
    #plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--',alpha=0.5)
    #plt.show()
    plt.title(r'Powerlaw, $n=10,\mu=1,k=1$', fontsize=18)

    plt.subplot(2, 2, 2)
    n = 10
    X = np.mat([[1 for x in range(n)], [math.log(1 + (x + 1) / n, math.e) for x in range(n)]]).T
    I = np.mat([1., 1.]).T
    theta1 = 10.
    theta2 = 1.

    s = generate_s(n, theta1, theta2)
    # print(s)
    expec1 = expectation([theta1, theta2], n)
    expec2 = expectation2([theta1, theta2], n)
    std = math.sqrt(Var([theta1, theta2], n))

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for y in range(n)]:
            continue
        xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, np.inf], [-30, 30]]))
        theta1_mle = xopt['x'][0]
        theta2_mle = xopt['x'][1]
        r = generate_s(n, theta1_mle, theta2_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    print(Kaastra_mean(s), Kaastra_var(s))
    # print(Max_mean(s),Max_var(s))

    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    # x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    # y = chi2fun(x, n - 2)
    # plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(0, 20, 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(Kaastra_mean(s) - 3 * math.sqrt(Kaastra_var(s)), Kaastra_mean(s) + 3 * math.sqrt(Kaastra_var(s)),
                  0.05)
    y = normfun(x, Kaastra_mean(s), math.sqrt(Kaastra_var(s)))
    plt.plot(x, y, color='blueviolet', label='Alg.2a')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color='green',linestyle='dashdot', label='Alg.2b')

    x = np.arange(expec2 - 3 * std, expec2 + 3 * std, 0.05)
    y = normfun(x, expec2, std)
    plt.plot(x, y, color='cyan', label='Alg.3a', marker='^',markevery=100)

    x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    y = normfun(x, expec1, std)
    plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4a')
    plt.xlabel('C-stat', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    # plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.3, 0.9))
    # plt.show()
    plt.title(r'Powerlaw, $n=10,\mu=10,k=1$', fontsize=18)

    plt.subplot(2, 2, 3)
    n = 100
    X = np.mat([[1 for x in range(n)], [math.log(1 + (x + 1) / n, math.e) for x in range(n)]]).T
    I = np.mat([1., 1.]).T
    theta1 = 1.
    theta2 = 1.

    s = generate_s(n, theta1, theta2)
    # print(s)
    expec1 = expectation([theta1, theta2], n)
    expec2 = expectation2([theta1, theta2], n)
    std = math.sqrt(Var([theta1, theta2], n))

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for y in range(n)]:
            continue
        xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, np.inf], [-30, 30]]))
        theta1_mle = xopt['x'][0]
        theta2_mle = xopt['x'][1]
        r = generate_s(n, theta1_mle, theta2_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    print(Kaastra_mean(s), Kaastra_var(s))
    # print(Max_mean(s),Max_var(s))

    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color='orange', label='Alg.1')

    #x = np.arange(0, 20, 0.05)
    #y = chi2fun(x, n - 2)
    #plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(Kaastra_mean(s) - 3 * math.sqrt(Kaastra_var(s)), Kaastra_mean(s) + 3 * math.sqrt(Kaastra_var(s)),
                  0.05)
    y = normfun(x, Kaastra_mean(s), math.sqrt(Kaastra_var(s)))
    plt.plot(x, y, color='blueviolet', label='Alg.2a')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color='green',linestyle='dashdot', label='Alg.2b')

    x = np.arange(expec2 - 3 * std, expec2 + 3 * std, 0.05)
    y = normfun(x, expec2, std)
    plt.plot(x, y, color='cyan', label='Alg.3a', marker='^',markevery=100)

    x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    y = normfun(x, expec1, std)
    plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4a')
    plt.xlabel('C-stat', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    #plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    # plt.show()
    plt.title(r'Powerlaw, $n=100,\mu=1,k=1$', fontsize=18)

    plt.subplot(2, 2, 4)
    n = 100
    X = np.mat([[1 for x in range(n)], [math.log(1 + (x + 1) / n, math.e) for x in range(n)]]).T
    I = np.mat([1., 1.]).T
    theta1 = 10.
    theta2 = 1.

    s = generate_s(n, theta1, theta2)
    # print(s)
    expec1 = expectation([theta1, theta2], n)
    expec2 = expectation2([theta1, theta2], n)
    std = math.sqrt(Var([theta1, theta2], n))

    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        if x == [0 for y in range(n)]:
            continue
        xopt = opt.minimize(LLF, [theta1, theta2], bounds=([[1e-5, np.inf], [-30, 30]]))
        theta1_mle = xopt['x'][0]
        theta2_mle = xopt['x'][1]
        r = generate_s(n, theta1_mle, theta2_mle)
        for j in range(n):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))

    print(Kaastra_mean(s), Kaastra_var(s))
    # print(Max_mean(s),Max_var(s))

    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color='orange', label='Alg.1')

    #x = np.arange(0, 20, 0.05)
    #y = chi2fun(x, n - 2)
    #plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(Kaastra_mean(s) - 3 * math.sqrt(Kaastra_var(s)), Kaastra_mean(s) + 3 * math.sqrt(Kaastra_var(s)),
                  0.05)
    y = normfun(x, Kaastra_mean(s), math.sqrt(Kaastra_var(s)))
    plt.plot(x, y, color='blueviolet', label='Alg.2a')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color='green',linestyle='dashdot', label='Alg.2b')

    x = np.arange(expec2 - 3 * std, expec2 + 3 * std, 0.05)
    y = normfun(x, expec2, std)
    plt.plot(x, y, color='cyan', label='Alg.3a', marker='^',markevery=100)

    x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    y = normfun(x, expec1, std)
    plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True, color='cornflowerblue', label='Alg.4a')
    plt.xlabel('C-stat', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    #plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    # plt.show()
    plt.title(r'Powerlaw, $n=100,\mu=10,k=1$', fontsize=18)

    plt.tight_layout()
    plt.savefig('test.pdf')
