from utilities import *

def LLF(theta,x,snull):
    if snull!='constant' and snull!='powerlaw' and snull!='exp':
        print("No such Likelihood Function!")
        return 0
    n = len(x)
    s = generate_s(n,theta,snull)
    return np.sum(s-x*np.log(s))


def LLF_grad(theta, x, snull):
    n = len(x)
    s = generate_s(n, theta, snull)

    # 计算公共项 [1 - x/s]
    common_term = 1 - x / (s + 1e-10)  # 添加小量避免除零

    # 根据不同 snull 类型计算 ds/d(theta)
    if snull == "exp":
        return _grad_exp(theta, common_term, n)
    elif snull == "powerlaw":
        return _grad_powerlaw(theta, common_term, n)
    elif snull == "constant":
        return _grad_constant(theta, common_term, n)
    elif snull == "brokenpowerlaw":
        return _grad_broken_powerlaw(theta, common_term, n)
    else:
        raise ValueError("Invalid snull type")


def _grad_exp(theta, common_term, n):
    theta1, theta2 = theta
    i = np.arange(n)  # [0, 1, ..., n-1]
    exp_term = np.exp(-theta2 * i / n)

    ds_dtheta1 = exp_term
    ds_dtheta2 = -theta1 * (i / n) * exp_term

    grad_theta1 = np.sum(common_term * ds_dtheta1)
    grad_theta2 = np.sum(common_term * ds_dtheta2)
    return np.array([grad_theta1, grad_theta2])


def _grad_powerlaw(theta, common_term, n):
    mu, slope = theta
    i_vals = np.arange(1, n + 1)  # [1, 2, ..., n]
    base = 1 + i_vals / n

    ds_dmu = base ** (-slope)
    ds_dslope = -mu * (base ** (-slope)) * np.log(base)

    grad_mu = np.sum(common_term * ds_dmu)
    grad_slope = np.sum(common_term * ds_dslope)
    return np.array([grad_mu, grad_slope])

def _grad_constant(theta, common_term, n):
    return np.array([np.sum(common_term)])


def _grad_broken_powerlaw(theta, common_term, n, loc=0.5):
    mu, slope1, slope2 = theta
    breakpoint = int(n * loc)
    i_part1 = np.arange(1, breakpoint + 1)
    i_part2 = np.arange(1, n - breakpoint + 1)

    # 前段梯度（类似 powerlaw）
    base1 = 1 + i_part1 / n
    ds_dmu_front = base1 ** (-slope1)
    ds_dslope1 = -mu * (base1 ** (-slope1)) * np.log(base1)

    # 后段梯度（依赖前段最后一个值）
    s_break = mu * (1 + breakpoint / n) ** (-slope1)
    base2 = 1 + i_part2 / (n - breakpoint)
    ds_dmu_back = (1 + breakpoint / n) ** (-slope1) * base2 ** (-slope2)
    ds_dslope1_back = -mu * np.log(1 + breakpoint / n) * (1 + breakpoint / n) ** (-slope1) * base2 ** (-slope2)
    ds_dslope2 = -s_break * base2 ** (-slope2) * np.log(base2)

    # 合并梯度
    grad_mu = np.sum(common_term[:breakpoint] * ds_dmu_front) + np.sum(common_term[breakpoint:] * ds_dmu_back)
    grad_slope1 = np.sum(common_term[:breakpoint] * ds_dslope1) + np.sum(common_term[breakpoint:] * ds_dslope1_back)
    grad_slope2 = np.sum(common_term[breakpoint:] * ds_dslope2)

    return np.array([grad_mu, grad_slope1, grad_slope2])

def design_mat(beta,n,snull):
    p=len(beta)
    if snull=='constant':
        return np.mat([1. for i in range(n)]).T
    elif snull=='powerlaw':
        return np.mat([[1 for x in range(n)], [math.log(1+(x + 1) / n, math.e) for x in range(n)]]).T
    elif snull=='exp':
        return np.mat([[1 for x in range(n)], [(x + 1) / n for x in range(n)]]).T
    else:
        print("Design Matrix Fail!")
        return 0