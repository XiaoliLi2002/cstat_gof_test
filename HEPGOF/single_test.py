from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test, Kaastra_M_test, bootstrap_normal,bootstrap_empirical, bootstrap_bias_correction, bootstrap_double
import uncon_oracle, uncon_plugin, uncon_theory, con_theory, modified_theory_test

def single_test(n,B,B1,B2,beta,strue,snull,alpha,iters,epsilon=1e-5,loc=0.5,strength=3,width=5):
    s=generate_s_true(n,beta,strue,snull,loc=loc,strength=strength,width=width) # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    pvalues=np.zeros((iters,11)) #totally 10 methods, if strue=snull: Type I Error; if strue!=snull: Power
    for i in range(int(iters)): # repetition
        print(f"Iteration {i + 1}")
        x=poisson_data(s)
        if np.all(np.abs(x) < 1e-5): # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B") # Get MLE
        betahat=xopt['x']
        #print(betahat)
        r = generate_s(n, betahat, snull) # null-distribution
        Cmin=Cashstat(x,r)

        # start test
        pvalues[i,0]= Wilks_Chi2_test.p_value_chi(Cmin,n-len(betahat)) #Alg.1
        pvalues[i,1]= Kaastra_M_test.KMtest(Cmin,r) #Alg.2a
        pvalues[i,2]= bootstrap_normal.bootstrap_asymptotic(Cmin, betahat, n, snull, B) #Alg.2b
        pvalues[i,3]= uncon_oracle.oracle_uncon_test(Cmin,beta, n, snull,)#oracle uncon test, cannot be used when test power -- if H_0 not true, what is the true beta?
        pvalues[i,4]= uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull)  # plugin uncon test
        pvalues[i,5]= uncon_theory.uncon_theory_test(Cmin,betahat, n, snull) #Alg.3a
        pvalues[i,6]= con_theory.con_theory_test(Cmin,betahat, n, snull) #Alg.3b
        pvalues[i,7]= bootstrap_empirical.bootstrap_test(Cmin,betahat, n, snull,B)#4a
        pvalues[i,8]= bootstrap_bias_correction.bootstrap_bias(Cmin,betahat,n,snull,B1,B2)#4b
        pvalues[i,9]= bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1,B2)#4c
        pvalues[i,10]= modified_theory_test.modified_theory_test(Cmin, betahat, n, snull)#(1-p/n)

    return pvalues    #Rejection Rate

def reject_rate(pvalues,alpha):
    reject = pvalues <= alpha
    return np.mean(reject,axis=0)

if __name__=="__main__":
    epsilon_error = 1e-5  # error-control, user decide

    n = 50  # number of bins  10,25,50,100
    B = B1 = B2 = 100  # bootstrap repetition times
    beta = np.array([1, 1])  # ground-truth beta*  0, 1, 2
    strue = 'brokenpowerlaw'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 3, 5]  # For broken-powerlaw and spectral line
    alpha = 0.1  # significance level  0.01-0.25
    iters = 1e4  # repetition times
    np.random.seed(42)  # random seed
    pvalues= single_test(n, B, B1, B2, beta, strue, snull, alpha, iters, epsilon=epsilon_error, loc=loc,
                         strength=strength, width=width) # recommend to save p-values
    rejections=reject_rate(pvalues,alpha)
    print(rejections)
    # Do not forget to store the result, like pd.to_cvs(...)

    # You may add another loop for different n, beta, strue/snull and alpha
