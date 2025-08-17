from Simulations.utilities.utilities import *
from Simulations.utilities.Likelihood_Design_mat import LLF, LLF_grad
import Simulations.utilities.Wilks_Chi2_test
import Simulations.utilities.uncon_plugin,  Simulations.utilities.con_theory, Simulations.utilities.bootstrap_double, Simulations.utilities.bootstrap_empirical
import os
import time

def single_test_with_single_bootstrap_pvalue(n,beta,strue,snull,iters,B=1000,epsilon=1e-5,loc=0.5,strength=3,width=5):
    inital_time=time.time()
    s = generate_s_true(n, beta, strue, snull, loc=loc, strength=strength,
                        width=width)  # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    pvalues_onesided=np.ones((iters,5))  # totally 4 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided = np.ones((iters, 5))  # totally 4 methods, if strue=snull: Type I Error; if strue!=snull: Power
    times=np.zeros((iters, 5))
    for i in range(int(iters)):  # repetition
        print(f"Iteration {i+1} executing...")
        if (i-3)%5==0:
            print(f"Iteration {i} finished!")
            print(f"Time used: {time.time()-inital_time}")
            print(f"Estimated total time: {(time.time()-inital_time)*iters/(i)}")
            print(f"Last iteration time used: {times[i-1]}\n")
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):  # x==0, always accept
            continue

        if snull=='constant':
            betahat=np.array([np.mean(x)])
        else:            
            bound = empirical_bounds(x, snull, epsilon)
            xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")  # Get MLE
            betahat = xopt['x']
        # print(betahat)
        r = generate_s(n, betahat, snull)  # null-distribution
        Cmin = Cashstat(x, r)

        # start test
        time_now=time.time()
        pvalues_onesided[i,0], pvalues_twosided[i,0], _, _=Simulations.utilities.Wilks_Chi2_test.p_value_chi(Cmin, n - len(betahat)) # Chisq test
        times[i,0]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,1], pvalues_twosided[i,1], _, _=Simulations.utilities.uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull)# plugin uncon test
        times[i,1]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,2], pvalues_twosided[i,2], _, _= Simulations.utilities.con_theory.con_theory_test(Cmin, betahat, n, snull) # Conditional test
        times[i,2]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,3], pvalues_twosided[i,3], _, _= Simulations.utilities.bootstrap_empirical.bootstrap_test(Cmin,betahat,n,snull,B=B)  # Empirical Bootstrap test
        times[i,3]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,4], pvalues_twosided[i,4]= Simulations.utilities.bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1=B,B2=B)  # Double Bootstrap test
        times[i,4]=time.time()-time_now

    return pvalues_onesided, pvalues_twosided, times


if __name__=="__main__":
    '''
    Setting:
    1) Sparse:
        constant vs spec:
            n=10; mu=1; k=1; strength= 3 or 5; loc=0.5, width=0.1*n or 0.2*n
        constant vs broken powerlaw:
            n=10; mu=5 or 10; k=1; k'=3; loc=0.5
    --> 2x2 figure6: Type I, Type I, Spec. power; brokenP power

    2) Very sparse:
        constant vs spec:
            n= 1e3 and 1e4; mu=1e-2; strength=0.1; width=0.01*n; loc=0.5
    --> 2x2 figure7 
    '''
    n = 100  # number of bins
    B = 300
    beta = np.array([1, 1])  # ground-truth beta*
    strue = 'powerlaw'  # true s : 'powerlaw'/ 'brokenpowerlaw'/ 'spectral_line'
    snull = 'powerlaw'  # s of H_0 : 'powerlaw' / 'constant'
    loc, strength, width = [0.5, 0.1, int(0.01*n)]  # For broken-powerlaw and spectral line
    iters = 100  # repetition times, suppose p=0.1. Then CI = +-0.01 (3k), +-0.02 (1k). p=0.25, then CI = +-0.015 (3k). p=0.5, CI = +-0.02( 3k)
    np.random.seed(0)  # random seed

    # significance level (0.01~0.25，step 0.01)
    alpha_values=np.linspace(0.01,0.25,25)

    params = {
        'n': n,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': iters
    }


    # Main
    print("Executing...")
    pvalues_onesided, pvalues_twosided, execution_time=single_test_with_single_bootstrap_pvalue(n,beta,strue,snull,int(iters),B=B,strength=strength,loc=loc,width=width)

    method_names=['Chisq', 'Plug_in', 'Cond', 'SingleB', 'DoubleB']

    # Time
    results = {method_names[i]: execution_time[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    # Change Column order
    df = df[method_names]
    # Set Save Directory （Path).
    save_dir = "results/data_double/time/"
    os.makedirs(save_dir, exist_ok=True)
    # To Excel
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # cat
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    # Set Excel format
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(Time) Successfully Saved Data to: {os.path.abspath(full_path)}")

    # one-sided test
    results = []
    for alphas in alpha_values:
        results.append({'alpha': alphas, **reject_rate(pvalues_onesided,alphas, method_names=method_names)})

    print("\n Computation finished. Now exporting...")

    # DataFrame
    df = pd.DataFrame(results)

    # colnames order
    df = df[['alpha']+method_names]

    # set dir
    save_dir = "results/data_double/onesided"
    os.makedirs(save_dir, exist_ok=True)
    # toExcel
    filename=generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    # set Excel format
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(One-sided) Successfully Saved Data to: {os.path.abspath(full_path)}")

    # two-sided test
    results = []
    for alphas in alpha_values:
        results.append({'alpha': alphas, **reject_rate(pvalues_twosided, alphas, method_names=method_names)})

    df = pd.DataFrame(results)

    df = df[['alpha']+method_names]

    save_dir = "results/data_double/twosided"
    os.makedirs(save_dir, exist_ok=True)
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # cat
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(Two-sided) Successfully Saved Data to: {os.path.abspath(full_path)}")