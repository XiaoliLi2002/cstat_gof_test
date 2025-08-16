from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test
import uncon_plugin,  con_theory, bootstrap_empirical
import os
import time
from HEPGOF.testfunc import generate_filename_xlsx, reject_rate

def single_test_with_single_bootstrap_pvalue(n,beta,strue,snull,iters,B=1000,epsilon=1e-5,loc=0.5,strength=3,width=5):
    inital_time=time.time()
    s = generate_s_true(n, beta, strue, snull, loc=loc, strength=strength,
                        width=width)  # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    pvalues_onesided=np.ones((iters,4))  # totally 4 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided = np.ones((iters, 4))  # totally 4 methods, if strue=snull: Type I Error; if strue!=snull: Power
    times=np.zeros((iters, 4))

    CashCritical_onesided = np.zeros((iters, 4))
    Width_twosided = np.zeros((iters, 4))
    significance_level=0.1

    for i in range(int(iters)):  # repetition

        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):  # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")  # Get MLE
        betahat = xopt['x']
        # print(betahat)
        r = generate_s(n, betahat, snull)  # null-distribution
        Cmin = Cashstat(x, r)

        if (i-5)%300==0:
            print(f"Iteration {i} finished!")
            print(f"Time used: {time.time()-inital_time}")
            print(f"Estimated total time: {(time.time()-inital_time)*iters/i}\n")
            print(f"Results from last iteration: beta={betahat}, Cmin={Cmin}, Onesided pvalues: {pvalues_onesided[i-1]}")


        # start test
        time_now=time.time()
        pvalues_onesided[i,0], pvalues_twosided[i,0], CashCritical_onesided[i,0], Width_twosided[i,0]\
            =Wilks_Chi2_test.p_value_chi(Cmin, n - len(betahat), alpha=significance_level) # Chisq test
        times[i,0]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,1], pvalues_twosided[i,1], CashCritical_onesided[i,1], Width_twosided[i,1]\
            =uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull, alpha=significance_level)# plugin uncon test
        times[i,1]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,2], pvalues_twosided[i,2], CashCritical_onesided[i,2], Width_twosided[i,2]\
            = con_theory.con_theory_test(Cmin, betahat, n, snull, alpha=significance_level) # Conditional test
        times[i,2]=time.time()-time_now

        time_now=time.time()
        pvalues_onesided[i,3], pvalues_twosided[i,3], CashCritical_onesided[i,3], Width_twosided[i,3]\
            = bootstrap_empirical.bootstrap_test(Cmin,betahat,n,snull,B=B, alpha=significance_level)  # Empirical Bootstrap test
        times[i,3]=time.time()-time_now
    return pvalues_onesided, pvalues_twosided, times, CashCritical_onesided, Width_twosided


if __name__=="__main__":
    '''
    Setting:
    n=50
    snull='powerlaw'
        1) Type I Error vs alpha:
            strue="powerlaw"
            mu=1, 2.5, 5, 10; k=1 --> 2x2 figure2
        2) Power vs alpha:
            strue="brokenP"
                mu=1, 2.5, 5, 10; k=1; k'=3; loc=0.5 --> 2x2 figure3
            strue="spec"
                mu vs strength= 1 vs 0.1, 0.1 vs 1, 5 vs 1, 1 vs 3; k=1; loc=0.5; width=0.1*n --> 2x2 figure4
        3) Power vs deviance:
            strue="brokenP"
                mu=5; k=1; k'={1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}; loc=0.5. 9 points 
            strue="spec"
                mu=5 (mu[loc]~3.3), strength= {0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 3.2, 4.0, 5, 6.4, 8, 10}; k=1; loc=0.5; width=0.1*n.  Take log. 12 points
            --> 1x2 figure5
        4) Re-bin effect:
            snull="powerlaw"
                k=1
                mu=0.01, 0.1, 1, 2
                n= 1000, 100, 10, 5
            strue="spec"
                strength= 0.025, 0.25, 2.5, 5
                width= 0.2*n
            --> 2x2 figure6 Type I; 2x2 figure 7 Power
        5) Coverage vs Width (C-critical):
            strue="powerlaw"
            mu = 0.1, 2.5
            n = 10, 25, 50, 100, 200, 400;
            
            n = 10, 50 (if not good, 25, 100)
            mu = 0.1, 0.25, 0.5, 1, 1.6, 2.5, 5
            
        6) 3x3 figure:
            mu= 0.25, 1, 5
            strength= mu/10 or 3mu
            col1: type1; col2: mu/10; col3: 3mu
    '''

    # params
    n = 500  # number of bins
    B = 300
    beta = np.array([0.25, 1])  # ground-truth beta*
    strue = 'powerlaw'  # true s : powerlaw/ brokenpowerlaw/ spectral_line
    snull = 'powerlaw'  # s of H_0 : powerlaw
    loc, strength, width = [0.5, 3, int(0.1*n)]  # For broken-powerlaw and spectral line
    iters = 3000  # repetition times, suppose p=0.1. Then CI = +-0.01 (3k), +-0.02 (1k). p=0.25, then CI = +-0.015 (3k). p=0.5, CI = +-0.02( 3k)
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
    pvalues_onesided, pvalues_twosided, execution_time, CashCritical_onesided, Width_twosided\
        =single_test_with_single_bootstrap_pvalue(n,beta,strue,snull,int(iters),B=B,strength=strength,loc=loc,width=width)

    method_names=['Chisq', 'Plug_in', 'Cond', 'SingleB']

    # Time
    results = {method_names[i]: execution_time[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    # Change Column order
    df = df[method_names]
    # Set Save Directory （Path).
    save_dir = "results/datanew/time/"
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

    # One-sided Critical Value
    results = {method_names[i]: CashCritical_onesided[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    # Change Column order
    df = df[method_names]
    # Set Save Directory （Path).
    save_dir = "results/datanew/CriticalValue/"
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
    print(f"(CriticalValue) Successfully Saved Data to: {os.path.abspath(full_path)}")

    # Two-sided width
    results = {method_names[i]: Width_twosided[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    # Change Column order
    df = df[method_names]
    # Set Save Directory （Path).
    save_dir = "results/datanew/Width/"
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
    print(f"(Width) Successfully Saved Data to: {os.path.abspath(full_path)}")

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
    save_dir = "results/datanew/onesided"
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

    save_dir = "results/datanew/twosided"
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