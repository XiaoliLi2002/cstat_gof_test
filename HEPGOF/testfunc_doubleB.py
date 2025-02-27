from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad
import bootstrap_double
import os
import time
from typing import Set, Any, Dict
import multiprocessing
from functools import partial


def generate_filename_csv(params: dict) -> str:
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
    return "results_" + "_".join(components) + ".csv"

def double_bootstrap_timed_one_iter(iteration: int, params: Dict[str, Any]) -> dict:
    epsilon=1e-5
    print(f"迭代 {iteration} 使用参数: {params}")
    n=params["n"]
    beta=params["beta"]
    strue=params["strue"]
    strength=params["strength"]
    snull=params["snull"]
    loc=params["loc"]
    width=params["width"]
    B1=B2=params["B"]

    s=generate_s_true(n,beta,strue,snull,loc,strength,width) # ground-truth
    x=poisson_data(s)
    if np.all(np.abs(x) < 1e-5): # x==0, always accept
        return {
        "pvalue_onesided": 1,
        "pvalue_twosided": 1,
        "time": 0,
        "iteration": iteration
    }
    bound = empirical_bounds(x, snull, epsilon)
    xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B") # Get MLE
    betahat=xopt['x']
    r = generate_s(n, betahat, snull) # null-distribution
    Cmin=Cashstat(x,r)
    print(f" betahat={betahat}, Cmin={Cmin}")

 # Time each test
    time1=time.time()
    pvalues_onesided, pvalues_twosided= bootstrap_double.double_boostrap(Cmin,betahat,n,snull,B1,B2)#4c
    times = time.time()-time1

    return {
        "pvalue_onesided": [pvalues_onesided],
        "pvalue_twosided": [pvalues_twosided],
        "time": [times],
        "iteration": iteration
    }   #p-values and times

def get_existing_iterations(params) -> Set[int]:
    """获取所有结果文件中已存在的迭代号"""
    existing = set()
    save_dir = "results/data/time"
    filename = generate_filename_csv(params)
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts and parts[0].isdigit():
                    existing.add(int(parts[0]))
    return existing

def save_single_result(params: dict, result: dict):
    """安全保存单个迭代的结果"""
    try:
        # 准备数据行（包含迭代号）
        onesided_line = f"{result['iteration']}," + ",".join(f"{x:.6f}" for x in result["pvalue_onesided"]) + "\n"
        twosided_line = f"{result['iteration']}," + ",".join(f"{x:.6f}" for x in result["pvalue_twosided"]) + "\n"
        time_line = f"{result['iteration']}," + ",".join(f"{x:.4f}" for x in result["time"]) + "\n"
        # 原子化写入操作
        for save_dir, content in [('results/data/onesided', onesided_line),
                                   ('results/data/twosided', twosided_line),
                                   ('results/data/time', time_line)]:
            filename = generate_filename_csv(params)
            path = os.path.join(save_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', buffering=1) as f:  # 行缓冲模式
                f.write(content)
    except Exception as e:
        print(f"保存迭代 {result['iteration']} 失败: {str(e)}")

def process_iteration(iteration: int, params: Dict[str, Any]):
    """处理单个迭代的完整流程"""
    try:
        result = double_bootstrap_timed_one_iter(iteration,params=params)
        save_single_result(params=params,result=result)
        print(f"完成迭代 {iteration}")
        return iteration
    except Exception as e:
        print(f"迭代 {iteration} 出错: {str(e)}")
        return None

if __name__=="__main__":
    # Set Parameters
    n = 25  # number of bins
    beta = np.array([5., 1.])  # ground-truth beta*
    B1 = B2 = B = 1000  # Bootstrap
    strue = 'powerlaw'  # true s
    snull = 'powerlaw'  # s of H_0
    loc, strength, width = [0.5, 3, 5]  # For broken-powerlaw and spectral line
    TOTAL_ITERATIONS = 1000  # repetition times

    np.random.seed(42)  # random seed

    params = {
        'n': n,
        'B': B,
        'beta': beta,
        'strue': strue,
        'snull': snull,
        'strength': strength,
        'iters': TOTAL_ITERATIONS,
        'loc': loc,
        'width': width,
    }

    # Initializing
    WORKERS = multiprocessing.cpu_count()
    existing_iterations = get_existing_iterations(params)

    # 准备任务列表
    all_iterations = set(range(TOTAL_ITERATIONS))
    remaining_iterations = sorted(list(all_iterations - existing_iterations))
    print(f"发现 {len(existing_iterations)} 个已完成的迭代，剩余 {len(remaining_iterations)} 个待处理")


    # 进度跟踪函数
    def track_progress(start_time, total, completed):
        if completed == 0:
            return
        elapsed = time.time() - start_time
        avg = elapsed / completed
        remaining = avg * (total - completed)
        print(f"进度: {completed}/{total} | 已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s")


    processor = partial(process_iteration, params=params)

    # 并行处理
    with multiprocessing.Pool(processes=WORKERS) as pool:
        start_time = time.time()
        processed_count = 0
        # 使用异步迭代器
        results = pool.imap_unordered(processor, remaining_iterations, chunksize=5)

        # 处理结果
        for i, result in enumerate(results, 1):
            if result is not None:
                processed_count += 1

            # 每10次或最后次打印进度
            if i % 10 == 0 or i == len(remaining_iterations):
                track_progress(start_time, len(remaining_iterations), i)

        print(f"\n所有迭代处理完成，有效完成数: {processed_count}")
        print(f"总耗时: {time.time() - start_time:.1f}秒")
        print(f"Params setting: {params}")