from utilities import math, poisson, np


def poisson_dis(mu,i):
    return poisson.pmf(i,mu=mu)

def Sigma_diag(s,i,Q,n,max, epsilon):
    x=kapa12(s[i],max, epsilon)
    for j in range(n):
        x-=kapa11(s[j],max, epsilon)*Q[j,i]*kapa03(s[i],max, epsilon)
    return x

def expectation(s,n,X,I,max, epsilon):
    V = np.diag([s[i] for i in range(n)])
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n,max, epsilon) for i in range(n)])
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    for i in range(n):
        E += kapa1(s[i],max, epsilon)
    return float(E)

def uncon_expectation(s,n,X,I,max, epsilon):
    E=0
    for j in range(n):
        E += kapa1(s[j],max, epsilon)
    return float(E)

def uncon_var(s,n,X,I,max, epsilon):
    var=0
    for i in range(n):
        var += kapa2(s[i],max, epsilon)
    return var

def qtheta(s,n,X,I,max, epsilon):
    V = np.diag([s[i] for i in range(n)])
    k_11 = np.mat([kapa11(s[i], max, epsilon) for i in range(n)]).T
    return (k_11.T * X * (X.T * V * X) ** (-1) * X.T * k_11)[0, 0]

def Var(s,n,X,I,max, epsilon):
    return uncon_var(s,n,X,I,max, epsilon)-qtheta(s,n,X,I,max, epsilon)

def kapa1(mu,max,epsilon):
    x = 0
    for k in range(max):
        if k == 0:
            next_term=-2 * (k - mu) * poisson_dis(mu, k)
        else:
            next_term = 2 * (k * math.log(k / mu, math.e) - k + mu) * poisson_dis(mu, k)
        x += next_term
        if next_term<epsilon and k>mu:
            return x
    return x

def kapa2(mu,max, epsilon):
    x = 0
    k_1=kapa1(mu,max, epsilon)
    for k in range(max):
        if k == 0:
            next_term= (-2 * (k - mu) - k_1) ** 2 * poisson_dis(mu, k)
        else:
            next_term= (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) ** 2 * poisson_dis(mu, k)
        x += next_term
        if next_term<epsilon and k>mu:
            return x
    return x

def kapa11(mu,max, epsilon):
    x = 0
    k_1 = kapa1(mu, max, epsilon)
    for k in range(max):
        if k == 0:
            next_term= (-2 * (k - mu) - k_1) * (k - mu) * poisson_dis(mu, k)
        else:
            next_term= (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu) * poisson_dis(mu, k)
        x += next_term
        if next_term<epsilon and k>mu:
            return x
    return x

def kapa12(mu,max, epsilon):
    x = 0
    k_1 = kapa1(mu, max, epsilon)
    for k in range(max):
        if k == 0:
            next_term= (-2 * (k - mu) - k_1) * (k - mu) ** 2 * poisson_dis(mu, k)
        else:
            next_term= (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu) ** 2 * poisson_dis(mu,k)
        x += next_term
        if next_term<epsilon and k>mu:
            return x
    return x

def kapa03(mu,max,epsilon):
    return mu
