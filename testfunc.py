from utilities import *
from Likelihood_Design_mat import *
import Wilks_Chi2_test, Kaastra_M_test, bootstrap_normal,bootstrap_empirical, bootstrap_bias_correction, bootstrap_double
import uncon_oracle, uncon_plugin, uncon_theory, con_theory
import os

# A list to store the timing information

def single_test(n,B,B1,B2,beta,strue,snull,alpha,iters,maximum=int(1e5),epsilon=1e-5,loc=0.5,strength=10,width=2):
    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    rejections=np.zeros(10) #totally 8 methods, if strue=snull: Type I Error; if strue!=snull: Power
    for i in range(iters): # repetition
        print(f"Iteration {i + 1}")
        x=poisson_data(s)
        if np.all(np.abs(x) < 1e-5): # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        #bound = [[epsilon, max(x)]] if snull == 'constant' else [[epsilon, max(x)], [-math.log(max(x)/(min(x)+epsilon),2), math.log(max(x)/(min(x)+epsilon),2)]]
        #xopt = opt.minimize(LLF, beta, args=(x, snull), bounds=bound, method="L-BFGS-B") # Get MLE
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
        if bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1,B2)<alpha: rejections[9]+=1 #4c

    return rejections/iters     #Rejection Rate


def single_test_timed(n,B,B1,B2,beta,strue,snull,alpha,iters,maximum=int(1e5),epsilon=1e-5,loc=0.5,strength=10,width=2):
    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    rejections=np.zeros(10) #totally 8 methods, if strue=snull: Type I Error; if strue!=snull: Power
    # A list to store the timing information
    execution_times = []
    for i in range(iters): # repetition
        print(f"Iteration {i + 1}")
        x=poisson_data(s)
        if np.all(np.abs(x) < 1e-5): # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        #bound = [[epsilon, max(x)]] if snull == 'constant' else [[epsilon, max(x)], [-math.log(max(x)/(min(x)+epsilon),2), math.log(max(x)/(min(x)+epsilon),2)]]
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B") # Get MLE
        #xopt = opt.minimize(LLF, beta, args=(x, snull), bounds=bound, method="L-BFGS-B") # Get MLE
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


def output_results_to_file(dir, result_all, n, B1, B2, beta, strue, snull, alpha, iters, timed = True):
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


