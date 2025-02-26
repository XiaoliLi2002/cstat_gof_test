from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test, bootstrap_normal,bootstrap_empirical
import uncon_oracle, uncon_plugin, uncon_theory, con_theory, modified_theory_test
import os
import time

def single_test_timed_no_double_bootstrap(n,beta,strue,snull,B=1000,iters=1000,epsilon=1e-5,loc=0.5,strength=10,width=2):
    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    pvalues_onesided=np.ones((iters,8)) #totally 8 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided=np.ones((iters,8))
    times=np.zeros((iters,8)) # Execution time

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

 # Time each test
        time1=time.time()
        pvalues_onesided[i,0], pvalues_twosided[i,0]= Wilks_Chi2_test.p_value_chi(Cmin,n-len(betahat)) #Alg.1
        times[i,0] = time.time()-time1

        time1=time.time()
        pvalues_onesided[i,1], pvalues_twosided[i,1]= (
            uncon_oracle.oracle_uncon_test(Cmin,beta, n, snull)) #oracle uncon test, cannot be used when test power -- if H_0 not true, what is the true beta?
        times[i,1] = time.time()-time1

        time1=time.time()
        pvalues_onesided[i,2], pvalues_twosided[i,2]= uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull)  # plugin uncond test
        times[i,2] = time.time()-time1

        time1=time.time()
        pvalues_onesided[i,3], pvalues_twosided[i,3]= uncon_theory.uncon_theory_test(Cmin,betahat, n, snull) #Alg.3a
        times[i,3] = time.time()-time1

        time1=time.time()
        pvalues_onesided[i,4], pvalues_twosided[i,4]= con_theory.con_theory_test(Cmin,betahat, n, snull) #Alg.3b
        times[i,4] = time.time()-time1

        time1 = time.time()
        pvalues_onesided[i, 5], pvalues_twosided[i, 5] = modified_theory_test.modified_theory_test(Cmin, betahat, n,
                                                                                                     snull)  # (1-p/n)
        times[i, 5] = time.time() - time1

        time1 = time.time()
        pvalues_onesided[i, 6], pvalues_twosided[i, 6] = bootstrap_normal.bootstrap_asymptotic(Cmin, betahat, n, snull,
                                                                                               B)  # Alg.2b
        times[i, 6] = time.time() - time1

        time1=time.time()
        pvalues_onesided[i,7], pvalues_twosided[i,7]= bootstrap_empirical.bootstrap_test(Cmin,betahat, n, snull,B)#4a
        times[i,7] = time.time()-time1

    return pvalues_onesided, pvalues_twosided, times    #p-values and ave. time

def reject_rate(pvalues, alpha, method_names):
    reject = pvalues <= alpha
    rejectrate=np.mean(reject,axis=0)
    return {method_names[i]: rejectrate[i] for i in range(len(method_names))}

def generate_filename_xlsx(params: dict) -> str:
    """生成包含参数的文件名"""
    components = []

    # 按参数顺序处理每个参数（保持你想要的顺序）
    for key in ['n', 'B', 'beta', 'strue', 'snull', 'strength', 'iters']:
        value = params[key]

        # 处理列表类型的参数（如beta）
        if isinstance(value, np.ndarray):  # 处理numpy数组
            components.append(f"{key}{'_'.join(map(str, value))}")
        elif isinstance(value, list):
            components.append(f"{key}{'_'.join(map(str, value))}")
        # 处理字符串类型的参数
        elif isinstance(value, str):
            components.append(f"{key}{value}")
        # 处理数值类型的参数
        else:
            components.append(f"{key}{value}")
    return "results_" + "_".join(components) + ".xlsx"

if __name__=="__main__":
    # Set Parameters
    n = 10  # number of bins
    beta = np.array([1., 1.])  # ground-truth beta*
    B = 100  # Bootstrap
    strue = 'spectral_line'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 5, 3]  # For broken-powerlaw and spectral line
    iters = 50  # repetition times


    np.random.seed(42)  # random seed


    params = {
        'n': n,
        'B': B,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': iters
    }
    method_names=['Chisq', 'Oracle', 'Plug_in', 'Uncond', 'Cond', 'Modified', 'Asymptotic_B', 'Empirical_B']

    # Calculating p-values
    print("Executing...")
    pvalues_onesided, pvalues_twosided, execution_time = single_test_timed_no_double_bootstrap(n,beta,strue,snull,
                                                                           B=B,iters=int(iters),strength=strength,loc=loc,width=width)
    print("\n Computation finished. Now exporting...")

    # Time
    results = {method_names[i]: execution_time[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    # Change Column order
    df = df[method_names]
    # Set Save Directory （Path). Please Create before running.
    save_dir = "results/data/time"

    # To Excel
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
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

    # P-values
    # One-sided
    results = {method_names[i]: pvalues_onesided[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    df = df[method_names]

    save_dir = "results/data/pvalues/onesided"
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(One-sided test) Successfully Saved Data to: {os.path.abspath(full_path)}")

    # Two-sided
    results = {method_names[i]: pvalues_twosided[0:, i] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    df = df[method_names]

    save_dir = "results/data/pvalues/twosided"
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(Two-sided test) Successfully Saved Data to: {os.path.abspath(full_path)}")

# Rejection rates
'''
    # Generate significance level (0.01到0.25，步长0.01)
    alpha_values=np.linspace(0.01,0.25,25)
    
    # One-sided test
    results = []
    for alphas in alpha_values:
        results.append({'alpha': alphas, **reject_rate(pvalues_onesided,alphas,method_names=method_names)})

    print("\n Computation finished. Now exporting...")
    df = pd.DataFrame(results)
    # Change Column order
    df = df[['alpha']+method_names]
    # Set Save Directory （Path). Please Create before running.
    save_dir = "results/data/onesided"

    # To Excel
    filename=generate_filename(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
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
    print(f"(One-sided test) Successfully Saved Data to: {os.path.abspath(full_path)}")

    # Two-sided test
    results = []
    for alphas in alpha_values:
        results.append({'alpha': alphas, **reject_rate(pvalues_onesided, alphas, method_names=method_names)})
    df = pd.DataFrame(results)
    df = df[['alpha']+method_names]

    save_dir = "results/data/twosided"

    filename = generate_filename(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(Two-sided test) Successfully Saved Data to: {os.path.abspath(full_path)}")
'''



