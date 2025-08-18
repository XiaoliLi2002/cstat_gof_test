from HEAGOF.utilities.utilities import *

def bootstrap_test(model, Cmin, thetahat,B=1000, alpha=0.1):
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
    s = model.generate_s(thetahat)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        xopt = opt.minimize(model.LLF, thetahat, args=(x),
                            bounds=model.bound)
        betahat=xopt['x']
        r = model.generate_s(betahat)
        C[i]=Cashstat(x,r)
    return np.mean(C >= Cmin), 2*min(np.mean(C <= Cmin), np.mean(C >= Cmin)), np.quantile(C,1-alpha), np.quantile(C,1-alpha/2)-np.quantile(C,alpha/2)
