from utilities import *
from cumulants import uncon_expectation, uncon_var
from Likelihood_Design_mat import design_mat

def oracle_uncon_test(x,betatrue,n,snull,maximum=int(1e5),epsilon=1e-5):
    s = generate_s(n, betatrue, snull)
    I = np.mat([1.0 for i in range(len(betatrue))]).T
    X = design_mat(betatrue, n, snull)
    return p_value_norm(x, uncon_expectation(s, n, X, I, maximum, epsilon),
                        math.sqrt(uncon_var(s, n, X, I, maximum, epsilon)))