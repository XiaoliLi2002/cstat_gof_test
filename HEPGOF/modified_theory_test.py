from utilities import *
from cumulants import uncon_expectation, Var
from Likelihood_Design_mat import design_mat

def modified_theory_test(x,beta,n,snull):  #Alg.3a
    s=generate_s(n,beta,snull)
    p=len(beta)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    X=design_mat(beta,n,snull)
    return p_value_norm(x,(1-p/n)*uncon_expectation(s,n,X,I),math.sqrt((1-p/n)*Var(s,n,X,I)))