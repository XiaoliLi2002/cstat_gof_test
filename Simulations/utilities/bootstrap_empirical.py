from Simulations.utilities.utilities import *
from Simulations.utilities.Likelihood_Design_mat import LLF, LLF_grad

def bootstrap_test(Cmin,beta,n,snull,B=1000, epsilon=1e-5, alpha=0.1):
    """
        Empirical Bootstrap test
        Compute the one-sided and two-sided p-values

        Args:
            Cmin: Cash statistics
            beta: estimator
            n: number of bins
            snull: null model H0
            B: bootstrap size (default 1000)
            alpha: significance level (=0.1)

        Returns:
            one-sided p-value
            two-sided p-value

            one-sided Critical Value
            two-sided width
    """
    C = np.zeros(B)
    s=generate_s(n,beta,snull)
    for i in range(B):
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
        C[i]=Cashstat(x,r)
    return np.mean(C >= Cmin), 2*min(np.mean(C <= Cmin), np.mean(C >= Cmin)), np.quantile(C,1-alpha), np.quantile(C,1-alpha/2)-np.quantile(C,alpha/2)
