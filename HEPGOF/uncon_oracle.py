from utilities import *
from cumulants import uncon_expectation, uncon_var
from Likelihood_Design_mat import design_mat


def oracle_uncon_test(x, betatrue, n, snull):
    s = generate_s(n, betatrue, snull)
    I = np.asmatrix(np.repeat(1.0,len(betatrue))).T
    X = design_mat(betatrue, n, snull)
    return p_value_norm(x, uncon_expectation(s, n, X, I),
                        math.sqrt(uncon_var(s, n, X, I)))