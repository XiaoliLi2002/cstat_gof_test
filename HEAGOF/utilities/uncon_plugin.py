import numpy as np

from HEAGOF.utilities.utilities import *
from HEAGOF.utilities.cumulants import uncon_expectation, uncon_var

def uncon_plugin_test(model,alpha=0.1):
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
    s = model.s_fitted
    I = np.asmatrix(np.repeat(1.0,len(model.thetahat))).T
    X = np.asmatrix(model.fitted_design_matrix)
    n = len(model.data)
    ave=uncon_expectation(s, n, X, I)
    std=math.sqrt(uncon_var(s, n, X, I))
    return (*p_value_norm(model.cashstat, ave,
                        std), ave+norm.isf(alpha)*std, 2*norm.isf(alpha/2)*std)
