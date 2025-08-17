from Simulations.utilities.utilities import *
from Simulations.utilities.cumulants import uncon_expectation, Var
from Simulations.utilities.Likelihood_Design_mat import design_mat

def uncon_theory_test(x,beta,n,snull):
    """
        Unconditional test with theoretical correction
        Compute the one-sided and two-sided p-values

        Args:
            x: data
            beta: estimator
            n: number of bins
            snull: null model H0

        Returns:
            one-sided p-value
            two-sided p-value
    """
    s=generate_s(n,beta,snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    X=design_mat(beta,n,snull)
    return p_value_norm(x,uncon_expectation(s,n,X,I),math.sqrt(Var(s,n,X,I)))