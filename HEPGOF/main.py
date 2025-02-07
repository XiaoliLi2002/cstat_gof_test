from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test, Kaastra_M_test, bootstrap_normal,bootstrap_empirical, bootstrap_bias_correction, bootstrap_double
import uncon_oracle, uncon_plugin, uncon_theory, con_theory

def single_test(n,B,B1,B2,beta,strue,snull,alpha,iters,maximum=int(1e5),epsilon=1e-5):
    s=generate_s_true(n,beta,strue,snull,loc=0.5,strength=.5,width=5) # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    rejections=np.zeros(10) #totally 10 methods, if strue=snull: Type I Error; if strue!=snull: Power
    for i in range(iters): # repetition
        print(f"Iteration {i + 1}")
        x=poisson_data(s)
        if np.all(np.abs(x) < 1e-5): # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B") # Get MLE
        betahat=xopt['x']
        print(betahat)
        r = generate_s(n, betahat, snull) # null-distribution
        Cmin=Cashstat(x,r)

        # start test
        if Wilks_Chi2_test.p_value_chi(Cmin,n-len(betahat))<alpha: #Alg.1
            rejections[0]+=1
        if Kaastra_M_test.KMtest(Cmin,r)<alpha: #Alg.2a
            rejections[1]+=1
        if bootstrap_normal.bootstrap_asymptotic(Cmin, betahat, n, snull, B)<alpha: #Alg.2b
            rejections[2]+=1
        if uncon_oracle.oracle_uncon_test(Cmin,beta, n, snull,maximum, epsilon)<alpha: rejections[3]+=1 #oracle uncon test, cannot be used when test power -- if H_0 not true, what is the true beta?
        if uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull, maximum, epsilon) < alpha: rejections[4] += 1  # plugin uncon test
        if uncon_theory.uncon_theory_test(Cmin,betahat, n, snull,maximum, epsilon)<alpha: rejections[5]+=1 #Alg.3a
        if con_theory.con_theory_test(Cmin,betahat, n, snull,maximum, epsilon)<alpha: rejections[6]+=1 #Alg.3b
        if bootstrap_empirical.bootstrap_test(Cmin,betahat, n, snull,B)<alpha: rejections[7]+=1 #4a
        if bootstrap_bias_correction.bootstrap_bias(Cmin,betahat,n,snull,B1,B2)<alpha: rejections[8]+=1 #4b
        #if bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1,B2)<alpha: rejections[9]+=1 #4c

    return rejections/iters     #Rejection Rate

epsilon_error=1e-5 # error-control, user decide
maximum_iter=int(1e5) # maximum iterations, user decide

n=10  # number of bins
B=B1=B2=100  # bootstrap repetition times
beta=np.array([8,0.1])  #ground-truth beta*
strue='brokenpowerlaw'  # true s
snull='powerlaw'  # s of H_0
alpha=0.1  # significance level
iters=100  # repetition times
result=single_test(n, B, B1, B2, beta, strue, snull, alpha, iters, maximum_iter, epsilon_error)
print(result)
# Do not forget to store the result, like pd.to_cvs(...)

# You may add another loop for different n, beta, strue/snull and alpha
