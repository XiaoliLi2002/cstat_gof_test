from mpmath import mp, exp, log, gamma, fmul, fabs
import pandas as pd
import time


def compute_moments(mu, epsilon=1e-30, max_terms=10 ** 5):
    """moments computing functions"""
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
        # f(Y)
        if k == 0:
            y_log_y = mp.mpf(0)
        else:
            y_log_y = k * log(k)
        f_k = 2 * (mu - k * log_mu - k + y_log_y)

        Yc_sq = (k - mu) ** 2

        # add
        term = current_p
        sum_f += term * f_k
        sum_f_sq += term * f_k ** 2
        sum_fY += term * f_k * k
        sum_fYc_sq += term * f_k * Yc_sq
        sum_Yc_sq += term * Yc_sq

        # check
        check = [
            abs(term * f_k) < epsilon * abs(sum_f),
            abs(term * f_k ** 2) < epsilon * abs(sum_f_sq),
            abs(term * f_k * k) < epsilon * abs(sum_fY),
            abs(term * f_k * Yc_sq) < epsilon * abs(sum_fYc_sq),
            abs(term * Yc_sq) < epsilon * abs(sum_Yc_sq)
        ]

        if all(check):
            consecutive_all_small += 1
            if consecutive_all_small >= 3:
                break
        else:
            consecutive_all_small = 0

        # next term
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
    """cumulants computing functions"""
    moments = compute_moments(mu)

    E_f = moments['E_f']
    E_f_sq = moments['E_f_sq']
    E_fY = moments['E_fY']
    E_fYc_sq = moments['E_fYc_sq']
    Var_Y = mu  # known variance

    return {
        'k1': float(E_f),
        'k2': float(E_f_sq - E_f ** 2),
        'k11': float(E_fY - E_f * mu),
        'k12': float(E_fYc_sq - E_f * Var_Y)
    }


if __name__ == '__main__':
    def print_progress(current):
        elapsed = time.time() - start_time
        remaining = elapsed / (current + 1) * (total - current - 1)
        print(f"\r进度: {current + 1}/{total} [用时: {elapsed:.1f}s, 剩余: {remaining:.1f}s]", end="")


    # generate mu (0~10，step 0.0001)
    mu_values = [round(x * 0.0001, 6) for x in range(100001)]


    results = []
    total = len(mu_values)
    start_time = time.time()

    # set dps
    mp.dps = 20
    # main calculation
    print("Start calculations...")
    for idx, mu in enumerate(mu_values):
        results.append({'mu': mu, **main_calculations(mu)})
        if idx % 100 == 0:
            print_progress(idx)

    print("\nFinished!")

    df = pd.DataFrame(results)

    df = df[['mu', 'k1', 'k2', 'k11', 'k12']]

    # to Excel
    writer = pd.ExcelWriter('poisson_results_0to10.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.10f")

    # set Excel format
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print("Successfully output data to poisson_results_0to10.xlsx")


    # generate mu (10~100，step 0.01)
    mu_values = [round( x * 0.01, 6) for x in range(10001)]


    results = []
    total = len(mu_values)
    start_time = time.time()


    mp.dps = 20

    print("Start calculations...")
    for idx, mu in enumerate(mu_values):
        results.append({'mu': mu, **main_calculations(mu)})
        if idx % 100 == 0:
            print_progress(idx)

    print("\nFinished!")

    df = pd.DataFrame(results)

    df = df[['mu', 'k1', 'k2', 'k11', 'k12']]

    writer = pd.ExcelWriter('poisson_results_0to100.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False, float_format="%.10f")

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_header = workbook.add_format({'bold': True, 'bg_color': '#FFFF00'})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format_header)
    worksheet.autofit()

    writer.close()
    print("Successfully output data to poisson_results_0to100.xlsx")