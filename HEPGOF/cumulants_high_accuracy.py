from mpmath import mp, exp, log, gamma, fmul, fabs
import pandas as pd
import time


def compute_moments(mu, epsilon=1e-30, max_terms=10 ** 5):
    """核心计算函数，返回原始矩量"""
    sum_f = mp.mpf(0)
    sum_f_sq = mp.mpf(0)
    sum_fY = mp.mpf(0)
    sum_fYc_sq = mp.mpf(0)
    sum_Yc_sq = mp.mpf(0)

    if mu == 0:
        return {k: mp.mpf(0) for k in ['E_f', 'E_f_sq', 'E_fY', 'E_fYc_sq', 'E_Yc_sq']}

    log_mu = log(mu)
    current_log_p = -mu
    current_p = exp(current_log_p)

    k = 0
    consecutive_all_small = 0

    while k <= max_terms:
        # 计算f(Y)
        if k == 0:
            y_log_y = mp.mpf(0)
        else:
            y_log_y = k * log(k)
        f_k = 2 * (mu - k * log_mu - k + y_log_y)

        Yc_sq = (k - mu) ** 2

        # 累加各项
        term = current_p
        sum_f += term * f_k
        sum_f_sq += term * f_k ** 2
        sum_fY += term * f_k * k
        sum_fYc_sq += term * f_k * Yc_sq
        sum_Yc_sq += term * Yc_sq

        # 动态精度检查
        check = [
            abs(term * f_k) < epsilon * abs(sum_f),
            abs(term * f_k ** 2) < epsilon * abs(sum_f_sq),
            abs(term * f_k * k) < epsilon * abs(sum_fY),
            abs(term * f_k * Yc_sq) < epsilon * abs(sum_fYc_sq),
            abs(term * Yc_sq) < epsilon * abs(sum_Yc_sq)
        ]

        if all(check):
            consecutive_all_small += 1
            if consecutive_all_small >= 3:  # 宽松停止条件以提高速度
                break
        else:
            consecutive_all_small = 0

        # 递推下一项
        k += 1
        current_log_p += log_mu - log(k)
        current_p = exp(current_log_p)

    return {
        'E_f': sum_f,
        'E_f_sq': sum_f_sq,
        'E_fY': sum_fY,
        'E_fYc_sq': sum_fYc_sq,
        'E_Yc_sq': sum_Yc_sq
    }


def main_calculations(mu):
    """主计算函数，返回格式化结果"""
    moments = compute_moments(mu)

    E_f = moments['E_f']
    E_f_sq = moments['E_f_sq']
    E_fY = moments['E_fY']
    E_fYc_sq = moments['E_fYc_sq']
    Var_Y = mu  # 泊松分布方差已知

    return {
        'k1': float(E_f),
        'k2': float(E_f_sq - E_f ** 2),
        'k11': float(E_fY - E_f * mu),
        'k12': float(E_fYc_sq - E_f * Var_Y)
    }


if __name__ == '__main__':
    # 生成μ值序列 (0到10，步长0.001)
    mu_values = [round(x * 0.001, 6) for x in range(20001)]  # 精确处理浮点数

    # 预分配结果存储
    results = []
    total = len(mu_values)
    start_time = time.time()


    # 进度打印函数
    def print_progress(current):
        elapsed = time.time() - start_time
        remaining = elapsed / (current + 1) * (total - current - 1)
        print(f"\r进度: {current + 1}/{total} [用时: {elapsed:.1f}s, 剩余: {remaining:.1f}s]", end="")

    # 设置高精度环境
    mp.dps = 20  # 20位精度已足够应对大多数科学计算需求
    print("初始化高精度计算环境...")
    # 主计算循环
    print("开始批量计算...")
    for idx, mu in enumerate(mu_values):
        results.append({'mu': mu, **main_calculations(mu)})
        if idx % 100 == 0:
            print_progress(idx)

    print("\n计算完成，开始导出数据...")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 优化列顺序
    df = df[['mu', 'k1', 'k2', 'k11', 'k12']]

    # 导出到Excel
    writer = pd.ExcelWriter('poisson_results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.10f")

    # 设置Excel格式
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print("数据已成功导出到 poisson_results.xlsx")