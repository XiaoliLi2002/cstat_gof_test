from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test, Kaastra_M_test, bootstrap_normal,bootstrap_empirical, bootstrap_bias_correction, bootstrap_double
import uncon_oracle, uncon_plugin, uncon_theory, con_theory, modified_theory_test
import os
import time

'''
def single_test_timed(n,B,B1,B2,beta,strue,snull,alpha,iters,maximum=int(1e5),epsilon=1e-5,loc=0.5,strength=10,width=2):
    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    pvalues_onesided=np.zeros((iters,11)) #totally 11 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided=np.zeros((iters,11))
    # A list to store the timing information
    execution_times = []
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
        wilks_result, wilks_time = time_function_call(Wilks_Chi2_test.p_value_chi, Cmin, n - len(betahat))
        execution_times.append(('Wilks test', wilks_time))
        if wilks_result < alpha:
            rejections[0] += 1

        km_result, km_time = time_function_call(Kaastra_M_test.KMtest, Cmin, r)
        execution_times.append(('Kaastra_M test', km_time))
        if km_result < alpha:
            rejections[1] += 1

        bootstrap_asymptotic_result, bootstrap_asymptotic_time = time_function_call(
            bootstrap_normal.bootstrap_asymptotic, Cmin, betahat, n, snull, B)
        execution_times.append(('Bootstrap Asymptotic test', bootstrap_asymptotic_time))
        if bootstrap_asymptotic_result < alpha:
            rejections[2] += 1

        oracle_result, oracle_time = time_function_call(
            uncon_oracle.oracle_uncon_test, Cmin, beta, n, snull, maximum, epsilon)
        execution_times.append(('Oracle Uncon test', oracle_time))
        if oracle_result < alpha:
            rejections[3] += 1

        plugin_result, plugin_time = time_function_call(
            uncon_plugin.uncon_plugin_test, Cmin, betahat, n, snull, maximum, epsilon)
        execution_times.append(('Uncon Plugin test', plugin_time))
        if plugin_result < alpha:
            rejections[4] += 1

        uncon_theory_result, uncon_theory_time = time_function_call(
            uncon_theory.uncon_theory_test, Cmin, betahat, n, snull, maximum, epsilon)
        execution_times.append(('Uncon Theory test', uncon_theory_time))
        if uncon_theory_result < alpha:
            rejections[5] += 1

        con_theory_result, con_theory_time = time_function_call(
            con_theory.con_theory_test, Cmin, betahat, n, snull, maximum, epsilon)
        execution_times.append(('Con Theory test', con_theory_time))
        if con_theory_result < alpha:
            rejections[6] += 1

        bootstrap_empirical_result, bootstrap_empirical_time = time_function_call(
            bootstrap_empirical.bootstrap_test, Cmin, betahat, n, snull, B)
        execution_times.append(('Bootstrap Empirical test', bootstrap_empirical_time))
        if bootstrap_empirical_result < alpha:
            rejections[7] += 1

        bootstrap_bias_result, bootstrap_bias_time = time_function_call(
            bootstrap_bias_correction.bootstrap_bias, Cmin, betahat, n, snull, B1, B2)
        execution_times.append(('Bootstrap Bias test', bootstrap_bias_time))
        if bootstrap_bias_result < alpha:
            rejections[8] += 1

        bootstrap_double_result, bootstrap_double_time = time_function_call(
            bootstrap_double.double_boostrap, Cmin, betahat, n, snull, B1, B2)
        execution_times.append(('Bootstrap Double test', bootstrap_double_time))
        if bootstrap_double_result < alpha:
            rejections[9] += 1

    return rejections / iters, execution_times
'''


def single_test_timed(n,beta,strue,snull,B=1000,iters=1000,epsilon=1e-5,loc=0.5,strength=10,width=2):
    B1=B2=B
    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    pvalues_onesided=np.zeros((iters,10)) #totally 10 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided=np.zeros((iters,10))
    times=np.zeros((iters,10)) # Execution time

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

        time1=time.time()
        pvalues_onesided[i,8], pvalues_twosided[i,8]= bootstrap_bias_correction.bootstrap_bias(Cmin,betahat,n,snull,B1,B2)#4b
        times[i,8] = time.time()-time1

        time1=time.time()
        pvalues_onesided[i,9], pvalues_twosided[i,9]= bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1,B2)#4c
        times[i,9] = time.time()-time1


    return pvalues_onesided, pvalues_twosided, np.mean(times,axis=0)    #p-values and ave. time

def reject_rate(pvalues, alpha, method_names):
    reject = pvalues <= alpha
    rejectrate=np.mean(reject,axis=0)
    return {method_names[i]: rejectrate[i] for i in range(len(method_names))}

'''
from collections import defaultdict
def summarize_computing_times(input_filename):
    # Initialize a dictionary to store the computing times for each method
    method_times = defaultdict(list)

    # Read the input file
    with open(input_filename, 'r') as file:
        for line in file:
            # Split the line into the method name and its computing time
            method_name, time_str = line.rsplit(': ', 1)

            # Convert the time from string to float and remove the 'seconds' unit
            time_in_seconds = float(time_str.replace(' seconds', ''))

            # Add the time to the corresponding method
            method_times[method_name].append(time_in_seconds)

    # Prepare the output filename by appending '_summary' before the file extension
    base, extension = os.path.splitext(input_filename)
    output_filename = f"{base}_summary{extension}"

    # Write the summarized computing times to the output text file
    with open(output_filename, 'w') as output_file:
        # Write the header
        output_file.write("Method,Time (seconds)\n")

        # Write the method times
        for method, times in method_times.items():
            times_str = ', '.join(f"{time:.5f}" for time in times)  # Formatting each time nicely
            output_file.write(f"{method}: {times_str}\n")


def output_results_to_file(dir, result_all, n, B1, B2, beta, strue, snull, alpha, iters, timed = True): #Assume here restuls=[df,time]
    if timed:
        result = result_all[0]
        execution_times = result_all[1]
    else:
        result = result_all
    
   # Format the beta array as a string
    beta_str = '_'.join(map(str, beta))
    # Construct file name using given parameters
    file_name = f"results_n{n}_B{B1}_{B2}_beta{beta_str}_strue{strue}_snull{snull}_alpha{alpha}_iters{iters}.txt"
    # Full path to the output file
    full_path = os.path.join(dir, file_name)
    # Output results to a file
    with open(full_path, 'w') as f:
        for value in result:
          f.write(f"{value}\n")
    print(f"Results written to {file_name}")
    
    if timed:
        file_name_time = f"exetime_n{n}_B{B1}_{B2}_beta{beta_str}_strue{strue}_snull{snull}_alpha{alpha}_iters{iters}.txt"
        # File path where timing results will be saved
        file_path = os.path.join(dir, file_name_time)
        # Write the timings to the text file
        with open(file_path, 'w') as f:
            for func_name, exec_time in execution_times:
               f.write(f'{func_name}: {exec_time:.6f} seconds\n')
        print(f"Execution times have been saved to '{file_path}'.")
        summarize_computing_times(file_path)
'''

def generate_filename(params: dict) -> str:
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
    beta = np.array([1, 1])  # ground-truth beta*
    B = 100  # Bootstrap
    strue = 'spectral_line'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 5, 3]  # For broken-powerlaw and spectral line
    iters = 50  # repetition times
    np.random.seed(42)  # random seed

    # Generate significance level (0.01到0.25，步长0.01)
    alpha_values=np.linspace(0.01,0.25,25)

    params = {
        'n': n,
        'B': B,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': iters
    }
    method_names=['Chisq', 'Oracle', 'Plug_in', 'Uncond', 'Cond', 'Modified', 'Asymptotic_B', 'Empirical_B', 'Bias_B', 'Double_B']

    # Calculating p-values
    print("Executing...")
    pvalues_onesided, pvalues_twosided, execution_time = single_test_timed(n,beta,strue,snull,
                                                                           B=B,iters=int(iters),strength=strength,loc=loc,width=width)

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

    # Time
    results={method_names[i]: [execution_time[i]] for i in range(len(method_names))}
    df = pd.DataFrame(results)
    df = df[method_names]

    save_dir = "results/data/time"

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
    print(f"(Time) Successfully Saved Data to: {os.path.abspath(full_path)}")
