from utilities import poisson, np, pd
from cumulants_high_accuracy import main_calculations


def poisson_dis(mu,i):
    return poisson.pmf(i,mu=mu)

def Sigma_diag(s,i,Q,n):
    x=kapa12(s[i])
    for j in range(n):
        x-=kapa11(s[j])*Q[j,i]*kapa03(s[i])
    return x

def expectation(s,n,X,I):
    V = np.diag([s[i] for i in range(n)])
    Q = X * (X.T * V * X) ** (-1) * X.T
    Sigma = np.diag([Sigma_diag(s, i, Q, n) for i in range(n)])
    E = -0.5*np.trace(X.T * Sigma * X * (X.T * V * X) ** (-1))
    for i in range(n):
        E += kapa1(s[i])
    return float(E)

def uncon_expectation(s,n,X,I):
    E=0
    for j in range(n):
        E += kapa1(s[j])
    return float(E)

def uncon_var(s,n,X,I):
    var=0
    for i in range(n):
        var += kapa2(s[i])
    return var

def qtheta(s,n,X,I):
    V = np.diag([s[i] for i in range(n)])
    k_11 = np.asmatrix([kapa11(s[i]) for i in range(n)]).T
    return (k_11.T * X * (X.T * V * X) ** (-1) * X.T * k_11)[0, 0]

def Var(s,n,X,I):
    return uncon_var(s,n,X,I)-qtheta(s,n,X,I)


#Calculate cumulants mathematically, cannot be used!!! Low numerical accuracy
'''
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
'''

#Calculate cumulants using high-accuracy computation, grid-values and fitted formula
df=pd.read_excel('poisson_cumulants_results.xlsx')
def kapa1(mu):
    if mu<-1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu<0.005:
        return main_calculations(mu)['k1']
    elif mu>=20:
        return 1+1/(6*mu)
    else:
        return df['k1'][int(mu*1e3)]

def kapa2(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k2']
    elif mu >= 20:
        return 2*kapa1(mu)**2
    else:
        return df['k2'][int(mu * 1e3)]

def kapa11(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k11']
    elif mu >= 20:
        return -1/(6*mu)
    else:
        return df['k11'][int(mu * 1e3)]

def kapa12(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k12']
    elif mu >= 20:
        return 2*mu
    else:
        return df['k12'][int(mu * 1e3)]

def kapa03(mu):
    return mu
