from utilities import *
from cumulants import expectation, Var
from Likelihood_Design_mat import design_mat

def con_theory_test(x,beta,n,snull, alpha=0.1):
    """
        Conditional test
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
    """
    s = generate_s(n, beta, snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    X = design_mat(beta, n, snull)
    ave=expectation(s,n,X,I)
    std=math.sqrt(Var(s,n,X,I))
    return (*p_value_norm(x,ave,std), ave+norm.isf(alpha)*std, 2*norm.isf(alpha/2)*std)