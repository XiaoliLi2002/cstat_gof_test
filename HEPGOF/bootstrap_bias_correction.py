from utilities import *
from bootstrap_empirical import bootstrap_test, LLF, LLF_grad

def bootstrap_bias(Cmin,beta,n,snull,B1=1000,B2=1000, epsilon=1e-5):  #Alg.4b
    theta_tilde=np.zeros(len(beta))
    s = generate_s(n, beta, snull)
    for i in range(B1):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        theta_tilde += xopt['x']
    theta_adj=2*beta-theta_tilde/B1
    if theta_adj[0]<1e-5:
        theta_adj[0]=0
    return bootstrap_test(Cmin,theta_adj,n,snull,B2)