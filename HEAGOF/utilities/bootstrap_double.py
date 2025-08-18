from HEAGOF.utilities.utilities import *
from HEAGOF.utilities.bootstrap_empirical import bootstrap_test

def double_boostrap(model,B1=1000,B2=1000, epsilon=1e-5):
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
    Cmin = model.cashstat
    s = model.s_fitted
    for i in range(B1):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        xopt = opt.minimize(model.LLF, model.initializer, args=(x),
                            bounds=model.bound)
        betahat=xopt['x']
        r = model.generate_s(betahat)
        C[i] = Cashstat(x, r)
        pvalue_onesided[i], pvalue_twosided[i], leftquantiles, rightquantiles = bootstrap_test(model,Cmin=C[i],thetahat=betahat,B=B2)
    p_onesided = np.mean(C >= Cmin)
    p_twosided = 2 * min(np.mean(C <= Cmin), np.mean(C >= Cmin))
    return np.mean(pvalue_onesided <= p_onesided), np.mean(pvalue_twosided <= p_twosided)