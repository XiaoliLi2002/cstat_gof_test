from Simulations.utilities.utilities import *
from Simulations.utilities.cumulants import uncon_expectation, uncon_var
from Simulations.utilities.Likelihood_Design_mat import design_mat

def uncon_plugin_test(x,beta,n,snull,alpha=0.1):
    """
        Unconditional test with plug-in estimator
        Compute the one-sided and two-sided p-values

        Args:
            x: data
            beta: estimator
            n: number of bins
            snull: null model H0
            alpha: significance level (=0.1)

        Returns:
            one-sided p-value
            two-sided p-value

            one-sided Critical Value
            two-sided width

    """
    s = generate_s(n, beta, snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    X = design_mat(beta, n, snull)
    ave=uncon_expectation(s, n, X, I)
    std=math.sqrt(uncon_var(s, n, X, I))
    return (*p_value_norm(x, ave,
                        std), ave+norm.isf(alpha)*std, 2*norm.isf(alpha/2)*std)
