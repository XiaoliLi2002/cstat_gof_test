from utilities import *
from cumulants import expectation, Var
from Likelihood_Design_mat import design_mat

def con_theory_test(x,beta,n,snull,maximum=int(1e5),epsilon=1e-5):  #Alg.3b
    s = generate_s(n, beta, snull)
    I = np.mat([1. for i in range(len(beta))]).T
    X = design_mat(beta, n, snull)
    return p_value_norm(x,expectation(s,n,X,I,maximum,epsilon),math.sqrt(Var(s,n,X,I,maximum,epsilon)))