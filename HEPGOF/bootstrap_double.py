from utilities import *
from bootstrap_empirical import bootstrap_test, LLF, LLF_grad

def double_boostrap(Cmin,beta,n,snull,B1=1000,B2=1000, epsilon=1e-5):  #Alg.4c
    C = np.zeros(B1)
    pvalue_onesided = np.zeros(B1)
    pvalue_twosided = np.zeros(B1)
    s = generate_s(n, beta, snull)
    for i in range(B1):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i] = Cashstat(x, r)
        pvalue_onesided[i], pvalue_twosided[i]=bootstrap_test(C[i],xopt['x'],n,snull,B2)
    p_onesided = np.mean(C >= Cmin)
    p_twosided = 2 * min(np.mean(C <= Cmin), np.mean(C >= Cmin))
    return np.mean(pvalue_onesided <= p_onesided), np.mean(pvalue_twosided <= p_twosided)