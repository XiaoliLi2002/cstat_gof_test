from Simulations.utilities.utilities import *
from Simulations.utilities.bootstrap_empirical import bootstrap_test, LLF, LLF_grad

def double_boostrap(Cmin,beta,n,snull,B1=1000,B2=1000, epsilon=1e-5):
    """
        Empirical double Bootstrap test
        Compute the one-sided and two-sided p-values

        Args:
            Cmin: Cash statistics
            beta: estimator
            n: number of bins
            snull: null model H0
            B: bootstrap size

        Returns:
            one-sided p-value
            two-sided p-value
    """
    C = np.zeros(B1)
    pvalue_onesided = np.zeros(B1)
    pvalue_twosided = np.zeros(B1)
    s = generate_s(n, beta, snull)
    for i in range(B1):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        if snull=='constant':
            betahat=np.array([np.mean(x)])
        else:
            bound = empirical_bounds(x, snull, epsilon)
            xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
            betahat=xopt['x']
        r = generate_s(n, betahat, snull)
        C[i] = Cashstat(x, r)
        pvalue_onesided[i], pvalue_twosided[i], leftquantiles, rightquantiles = bootstrap_test(C[i],betahat,n,snull,B2)
    p_onesided = np.mean(C >= Cmin)
    p_twosided = 2 * min(np.mean(C <= Cmin), np.mean(C >= Cmin))
    return np.mean(pvalue_onesided <= p_onesided), np.mean(pvalue_twosided <= p_twosided)