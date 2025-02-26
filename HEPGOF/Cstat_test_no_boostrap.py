from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test
import uncon_oracle, uncon_plugin, uncon_theory, con_theory, modified_theory_test
import os
from testfunc import generate_filename_xlsx, reject_rate

def single_test_no_bootstrap_pvalue(n,beta,strue,snull,iters,epsilon=1e-5,loc=0.5,strength=3,width=5):
    s = generate_s_true(n, beta, strue, snull, loc=loc, strength=strength,
                        width=width)  # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    pvalues_onesided=np.zeros((iters,6))  # totally 6 methods, if strue=snull: Type I Error; if strue!=snull: Power
    pvalues_twosided = np.zeros((iters, 6))  # totally 6 methods, if strue=snull: Type I Error; if strue!=snull: Power
    for i in range(int(iters)):  # repetition
        if i%1000==0:
            print(f"Iteration {i + 1}")
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):  # x==0, always accept
            continue
        bound = empirical_bounds(x, snull, epsilon)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")  # Get MLE
        betahat = xopt['x']
        # print(betahat)
        r = generate_s(n, betahat, snull)  # null-distribution
        Cmin = Cashstat(x, r)

        # start test
        pvalues_onesided[i,0], pvalues_twosided[i,0]=Wilks_Chi2_test.p_value_chi(Cmin, n - len(betahat))
        pvalues_onesided[i,1], pvalues_twosided[i,1]=uncon_oracle.oracle_uncon_test(Cmin,beta, n, snull)#oracle uncon test, cannot be used when test power -- if H_0 not true, what is the true beta?
        pvalues_onesided[i,2], pvalues_twosided[i,2]=uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull)# plugin uncon test
        pvalues_onesided[i,3], pvalues_twosided[i,3]=uncon_theory.uncon_theory_test(Cmin, betahat, n, snull) # Alg.3a
        pvalues_onesided[i,4], pvalues_twosided[i,4]= con_theory.con_theory_test(Cmin, betahat, n, snull) # Alg.3b
        pvalues_onesided[i,5], pvalues_twosided[i,5]= modified_theory_test.modified_theory_test(Cmin, betahat, n, snull)  # (1-p/n)
    return pvalues_onesided, pvalues_twosided


if __name__=="__main__":
    n = 50  # number of bins  10,25,50,100
    beta = np.array([10, 1])  # ground-truth beta* {1,2.5,5,10} x {0, 1, 2}
    strue = 'spectral_line'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 2, 5]  # For broken-powerlaw and spectral line
    iters = 1e4  # repetition times
    np.random.seed(42)  # random seed

    # 生成significance level (0.01到0.25，步长0.01)
    alpha_values=np.linspace(0.01,0.25,25)

    params = {
        'n': n,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': iters
    }


    # 主计算循环
    print("Executing...")
    pvalues_onesided, pvalues_twosided=single_test_no_bootstrap_pvalue(n,beta,strue,snull,int(iters),strength=strength,loc=loc,width=width)

    method_names=['Chisq', 'Oracle', 'Plug_in', 'Uncond','Cond', 'Modified']
    # one-sided test
    results = []
    for alphas in alpha_values:
        results.append({'alpha': alphas, **reject_rate(pvalues_onesided,alphas, method_names=method_names)})

    print("\n Computation finished. Now exporting...")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 优化列顺序
    df = df[['alpha']+method_names]

    # 定义保存目录（相对路径）
    save_dir = "results/data/onesided"

    # 导出到Excel
    filename=generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    # 设置Excel格式
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
        results.append({'alpha': alphas, **reject_rate(pvalues_onesided, alphas, method_names=method_names)})

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 优化列顺序
    df = df[['alpha']+method_names]

    # 定义保存目录（相对路径）
    save_dir = "results/data/twosided"

    # 导出到Excel
    filename = generate_filename_xlsx(params)
    full_path = os.path.join(save_dir, filename)  # 跨平台路径拼接
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.5f")

    # 设置Excel格式
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print(f"(Two-sided) Successfully Saved Data to: {os.path.abspath(full_path)}")