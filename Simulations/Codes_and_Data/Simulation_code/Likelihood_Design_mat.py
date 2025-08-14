from utilities import *

def LLF(theta,x,snull):
    """
    Compute the likelihood function

    Args:
        theta: parameter
        x: data
        snull: null model H0

    Returns:
        the likelihood
    """
    if snull!='constant' and snull!='powerlaw' and snull!='exp':
        print("No such Likelihood Function!")
        return 0
    n = len(x)
    s = generate_s(n,theta,snull)
    return np.sum(s-x*np.log(s))


def LLF_grad(theta, x, snull):
    """
    Compute the gradient of the likelihood function

    Args:
        theta: parameter
        x: data
        snull: null model H0

    Returns:
        the gradient of the likelihood
    """
    n = len(x)
    s = generate_s(n, theta, snull)

    # compute common term [1 - x/s]
    common_term = 1 - x / (s + 1e-10)

    # compute ds/d(theta) based on different null model H0
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

    # gradient before loc
    base1 = 1 + i_part1 / n
    ds_dmu_front = base1 ** (-slope1)
    ds_dslope1 = -mu * (base1 ** (-slope1)) * np.log(base1)

    # gradient after loc
    s_break = mu * (1 + breakpoint / n) ** (-slope1)
    base2 = 1 + i_part2 / (n - breakpoint)
    ds_dmu_back = (1 + breakpoint / n) ** (-slope1) * base2 ** (-slope2)
    ds_dslope1_back = -mu * np.log(1 + breakpoint / n) * (1 + breakpoint / n) ** (-slope1) * base2 ** (-slope2)
    ds_dslope2 = -s_break * base2 ** (-slope2) * np.log(base2)

    # Combine
    grad_mu = np.sum(common_term[:breakpoint] * ds_dmu_front) + np.sum(common_term[breakpoint:] * ds_dmu_back)
    grad_slope1 = np.sum(common_term[:breakpoint] * ds_dslope1) + np.sum(common_term[breakpoint:] * ds_dslope1_back)
    grad_slope2 = np.sum(common_term[breakpoint:] * ds_dslope2)

    return np.array([grad_mu, grad_slope1, grad_slope2])

'''
def design_mat(beta,n,snull):
    """
    Compute the design matrix under different null models H0

    Args:
        beta: parameters
        n: number of bins
        snull: null model H0

    Returns:
        Design matrix
    """
    p=len(beta)
    if snull=='constant':
        return np.ones((n,1))
    elif snull=='powerlaw':
        return np.asmatrix([[1 for x in range(n)], [math.log(1+(x + 1) / n, math.e) for x in range(n)]]).T
    elif snull=='exp':
        return np.asmatrix([[1 for x in range(n)], [(x + 1) / n for x in range(n)]]).T
    else:
        print("Design Matrix Fail!")
        return 0

'''
def design_mat(beta, n, snull):
    """
    Compute the design matrix under different null models H0.

    Args:
        beta (array-like): parameter vector (only its length may determine number of columns).
        n (int): number of bins (rows in the design matrix).
        snull (str): one of {'constant', 'powerlaw', 'exp'}.

    Returns:
        X (np.ndarray): an (n × p0) design matrix, where p0 depends on snull:
            - 'constant':      p0 = 1
            - 'powerlaw', 'exp': p0 = 2

    Raises:
        ValueError: if `snull` is not recognized.
    """
    snull = snull.lower()
    # always include intercept
    intercept = np.ones(n)

    if snull == 'constant':
        # Only intercept
        return intercept.reshape(n, 1)

    # build a normalized index vector x = (1, 2, ..., n) / n
    x = np.arange(1, n + 1) / n

    if snull == 'powerlaw':
        # second column = log(1 + x)
        feature = np.log(1 + x)
    elif snull == 'exp':
        # second column = x linearly
        feature = x
    else:
        raise ValueError(f"Unknown null model '{snull}'. Expected 'constant', 'powerlaw', or 'exp'.")

    # stack intercept and feature into (n, 2)
    return np.asmatrix(np.column_stack([intercept, feature]))
    
