from HEAGOF.utilities.utilities import *
from HEAGOF.utilities.cumulants import expectation, expectation_high_order, Var

def con_theory_test(model, alpha=0.1):
    """
        Conditional test with first order expansion
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

            Estimated mean
            Estimated variance
    """
    s = model.s_fitted
    I = np.asmatrix(np.repeat(1.0, len(model.thetahat))).T
    X = np.asmatrix(model.fitted_design_matrix)
    n,p=X.shape
    ave=expectation(s,n,X,I)
    std=math.sqrt(Var(s,n,X,I))
    return (*p_value_norm(model.cashstat,ave,std), ave+norm.isf(alpha)*std, 2*norm.isf(alpha/2)*std, ave, std**2)

def con_theory_test_high_order(model, alpha=0.1):
    """
        Conditional test with high order expansion
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

            Estimated mean
            Estimated variance
    """
    s = model.s_fitted
    I = np.asmatrix(np.repeat(1.0, len(model.thetahat))).T
    X = np.asmatrix(model.fitted_design_matrix)
    n,p=X.shape
    ave=expectation_high_order(s,n,X,I)
    std=math.sqrt(Var(s,n,X,I))
    return (*p_value_norm(model.cashstat,ave,std), ave+norm.isf(alpha)*std, 2*norm.isf(alpha/2)*std, ave, std**2)