from utilities import *
from cumulants import expectation, Var
from Likelihood_Design_mat import design_mat

def con_theory_test(x,beta,n,snull):  #Alg.3b
    s = generate_s(n, beta, snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    X = design_mat(beta, n, snull)
    return p_value_norm(x,expectation(s,n,X,I),math.sqrt(Var(s,n,X,I)))