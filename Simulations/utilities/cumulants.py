from Simulations.utilities.utilities import poisson, np, pd
from Simulations.utilities.cumulants_high_accuracy import main_calculations
import os

def poisson_dis(mu,i):
    """
    Compute poisson pmf at x=i with parameter mu
    """
    return poisson.pmf(i,mu=mu)

def Sigma_diag(s,i,Q,n):
    """
    Compute the diagonal of Sigma

    Args:
        s: expected counts
        i: location
        Q: matrix Q
        n: number of bins

    Returns:
        the i-th diagonal term of Sigma
    """
    x=kapa12(s[i])
    for j in range(n):
        x-=kapa11(s[j])*Q[j,i]*kapa03(s[i])
    return x

def Sigma_compute(s, Q):
    """
    Compute the full Sigma = diag(σ_i) efficiently.

    Args:
        s (array-like, shape (n,)): expected counts
        Q (ndarray, shape (n, n)): matrix Q

    Returns:
        Sigma (ndarray, shape (n, n)): diagonal matrix whose i-th entry is
            kapa12(s[i]) - kapa03(s[i]) * sum_j [ kapa11(s[j]) * Q[j, i] ].
    """
    Q=np.array(Q)
    s = np.asarray(s)
    
    kapa12_s = np.vectorize(kapa12)(s)
    kapa03_s = np.vectorize(kapa03)(s)
    kapa11_s = np.vectorize(kapa11)(s)

    
    # sums[i] = ∑_j kapa11_s[j] * Q[j, i]
    sums = Q.T @ kapa11_s    # shape (n,)

    
    diag = kapa12_s - kapa03_s * sums

    return np.diag(diag)

def expectation(s,n,X,I):
    """
    Compute the conditional expectation

    Args:
        s (array-like, shape (n,)): expected counts
        n: number of bins
        X: design matrix
        I: identity matrix

    Returns:
        conditional expectation
    """
    V = np.diag([s[i] for i in range(n)])
    Q = X @ (X.T @ V @ X) ** (-1) @ X.T
    Sigma = Sigma_compute(s,Q)
    E = -0.5*np.trace(X.T @ Sigma @ X @ (X.T @ V @ X) ** (-1))
    for i in range(n):
        E += kapa1(s[i])
    return float(E)

def uncon_expectation(s,n,X,I):
    """
    Compute the unconditional expectation

    Args:
        s (array-like, shape (n,)): expected counts
        n: number of bins
        X: design matrix
        I: identity matrix

    Returns:
        unconditional expectation
    """
    E=0
    n,p=X.shape
    for j in range(n):
        E += kapa1(s[j])
    return float(E)

def uncon_expectation_highorder(s,n,X,I):
    """
        Compute the unconditional expectation using high-order approximation

        Args:
            s (array-like, shape (n,)): expected counts
            n: number of bins
            X: design matrix
            I: identity matrix

        Returns:
            unconditional expectation
        """
    E = 0
    n, p = X.shape
    for j in range(n):
        E += kapa1(s[j])
    return max(float(E)-p,0)

def uncon_var(s,n,X,I):
    """
    Compute the unconditional variance

    Args:
        s (array-like, shape (n,)): expected counts
        n: number of bins
        X: design matrix
        I: identity matrix

    Returns:
        unconditional variance
    """
    var=0
    for i in range(n):
        var += kapa2(s[i])
    return var

def qtheta(s,n,X,I):
    """
    Compute the correction term for variance Q(theta)

    Args:
        s (array-like, shape (n,)): expected counts
        n: number of bins
        X: design matrix
        I: identity matrix

    Returns:
        Q(theta)
    """
    V = np.diag([s[i] for i in range(n)])
    k_11 = np.matrix(np.vectorize(kapa11)(s)).T
    return (k_11.T @ X @ (X.T @ V @ X) ** (-1) @ X.T @ k_11)[0, 0]

def Var(s,n,X,I):
    """
    Compute the conditional variance

    Args:
        s (array-like, shape (n,)): expected counts
        n: number of bins
        X: design matrix
        I: identity matrix

    Returns:
        conditional variance
    """
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..","utilities","poisson_cumulants_results.xlsx",)
data_path = os.path.abspath(data_path)
df1=pd.read_excel(data_path,sheet_name="0~10") # gap: 0.0001
df2=pd.read_excel(data_path,sheet_name="0~20") # gap: 0.001
df3=pd.read_excel(data_path,sheet_name="0~100") # gap: 0.01

def kapa1(mu):
    if mu < -1e-5: # mu < 0
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:  # mu 0 ~ 0.005
        return main_calculations(mu)['k1']
    elif mu < 10.0: # mu 0.005 ~ 10, gap 0.0001
        return df1['k1'][int(mu*1e4)]
    elif mu < 20.0: # mu 10 ~ 20, gap 0.001
        return df2['k1'][int(mu * 1e3)]
    elif mu < 100.0: # mu 20 ~ 100, gap 0.01
        return df3['k1'][int(mu * 1e2)]
    else: # mu >= 100
        return 1+1/(6*mu)

def kapa2(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k2']
    elif mu < 10.0: # mu 0.005 ~ 10, gap 0.0001
        return df1['k2'][int(mu*1e4)]
    elif mu < 20.0: # mu 10 ~ 20, gap 0.001
        return df2['k2'][int(mu * 1e3)]
    elif mu < 100.0: # mu 20 ~ 100, gap 0.01
        return df3['k2'][int(mu * 1e2)]
    else: # mu >= 100
        return 2*kapa1(mu)**2

def kapa11(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k11']
    elif mu < 10.0: # mu 0.005 ~ 10, gap 0.0001
        return df1['k11'][int(mu*1e4)]
    elif mu < 20.0: # mu 10 ~ 20, gap 0.001
        return df2['k11'][int(mu * 1e3)]
    elif mu < 100.0: # mu 20 ~ 100, gap 0.01
        return df3['k11'][int(mu * 1e2)]
    else: # mu >= 100
        return -1/(6*mu)


def kapa12(mu):
    if mu < -1e-5:
        print('Warning: Invalid mu < 0!')
        return 0
    elif mu < 0.005:
        return main_calculations(mu)['k12']
    elif mu < 10.0: # mu 0.005 ~ 10, gap 0.0001
        return df1['k12'][int(mu*1e4)]
    elif mu < 20.0: # mu 10 ~ 20, gap 0.001
        return df2['k12'][int(mu * 1e3)]
    elif mu < 100.0: # mu 20 ~ 100, gap 0.01
        return df3['k12'][int(mu * 1e2)]
    else: # mu >= 100
        return 2*mu

def kapa03(mu):
    return mu

