from Simulations.utilities.utilities import *
from Simulations.utilities.Likelihood_Design_mat import LLF, LLF_grad

def bootstrap_asymptotic(Cmin,beta,n,snull,B=1000, epsilon=1e-5):
    """
        Bootstrap test with normal approximation
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
    return p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C))