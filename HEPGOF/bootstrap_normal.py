from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad

def bootstrap_asymptotic(Cmin,beta,n,snull,B=1000, epsilon=1e-5):  #Alg.2b
    C = np.zeros(B)
    s=generate_s(n,beta,snull)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i]=Cashstat(x,r)
    return p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C))