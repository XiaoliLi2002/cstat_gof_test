from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import Wilks_Chi2_test
import uncon_oracle, uncon_plugin, uncon_theory, con_theory, modified_theory_test
import time
import os


def single_test_no_bootstrap_pvalue(n,beta,strue,snull,iters,epsilon=1e-5,loc=0.5,strength=3,width=5):
    s = generate_s_true(n, beta, strue, snull, loc=loc, strength=strength,
                        width=width)  # ground-truth. loc, strength & width are used for brokenpowerlaw, spectral line.
    print(s)
    pvalues=np.zeros((iters,6))  # totally 6 methods, if strue=snull: Type I Error; if strue!=snull: Power
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
        pvalues[i,0]=Wilks_Chi2_test.p_value_chi(Cmin, n - len(betahat))
        pvalues[i,1]=uncon_oracle.oracle_uncon_test(Cmin,beta, n, snull)#oracle uncon test, cannot be used when test power -- if H_0 not true, what is the true beta?
        pvalues[i,2]=uncon_plugin.uncon_plugin_test(Cmin, betahat, n, snull)# plugin uncon test
        pvalues[i,3]=uncon_theory.uncon_theory_test(Cmin, betahat, n, snull) # Alg.3a
        pvalues[i,4]= con_theory.con_theory_test(Cmin, betahat, n, snull) # Alg.3b
        pvalues[i,5]= modified_theory_test.modified_theory_test(Cmin, betahat, n, snull)  # (1-p/n)
    return pvalues

def reject_rate(pvalues,alpha):
    reject=pvalues<=alpha
    return {
        'Chisq': np.mean(reject[0:,0]),
        'Oracle': np.mean(reject[0:,1]),
        'Plug_in': np.mean(reject[0:,2]),
        'Uncond': np.mean(reject[0:,3]),
        'Cond': np.mean(reject[0:,4]),
        'Modified': np.mean(reject[0:,5]),
    }


def generate_filename(params: dict) -> str:
    """生成包含参数的文件名"""
    components = []

    # 按参数顺序处理每个参数（保持你想要的顺序）
    for key in ['n', 'beta', 'strue', 'snull', 'strength', 'iters']:
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
    n = 100  # number of bins  10,25,50,100
    beta = np.array([5, 1])  # ground-truth beta* {1,5,10} x {0, 1, 2}
    strue = 'brokenpowerlaw'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 3, 5]  # For broken-powerlaw and spectral line
    iters = 1e4  # repetition times
    np.random.seed(42)  # random seed

    # 生成significance level (0.01到0.25，步长0.01)
    #alpha_values = np.linspace(0.1, 0.1, 1)
    alpha_values=np.linspace(0.01,0.25,25)

    params = {
        'n': n,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': iters
    }

    # 定义保存目录（相对路径）
    save_dir = "results/data"

    # 预分配结果存储
    results = []
    total = len(alpha_values)
    start_time = time.time()


    # 进度打印函数
    def print_progress(current):
        elapsed = time.time() - start_time
        remaining = elapsed / (current + 1) * (total - current - 1)
        print(f"\r进度: {current + 1}/{total} [用时: {elapsed:.1f}s, 剩余: {remaining:.1f}s]", end="")


    # 主计算循环
    print("Executing...")
    pvalues=single_test_no_bootstrap_pvalue(n,beta,strue,snull,int(iters),strength=strength,loc=loc,width=width)
    for idx, alphas in enumerate(alpha_values):
        results.append({'alpha': alphas, **reject_rate(pvalues,alphas)})
        #if idx % 1 == 0:
        #    print_progress(idx)

    print("\n计算完成，开始导出数据...")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 优化列顺序
    df = df[['alpha', 'Chisq', 'Oracle', 'Plug_in', 'Uncond','Cond', 'Modified']]

    # 导出到Excel
    filename=generate_filename(params)
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
    print(f"Successfully Saved Data to: {os.path.abspath(full_path)}")