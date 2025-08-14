from utilities import *
from Likelihood_Design_mat import LLF, LLF_grad, design_mat
import matplotlib.pyplot as plt
from cumulants import uncon_expectation, uncon_var
from Kaastra_M_test import Max_mean, Max_var
import scipy

#np.set_printoptions(threshold=np.inf)
plt.figure(dpi=300,figsize=(18,12))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
#plt.rcParams['axes.linewidth'] = 2.0

def chi2fun(x,n):
    pdf=1/(2**(n/2)*scipy.special.gamma(n/2))*x**(n/2-1)*math.e**(-x/2)
    return pdf

B=2000
np.random.seed(42)
iters=1
strue='powerlaw'
snull='powerlaw'

palette = ["#F0E442",# Yellow
        "#CC79A7",  # Orange
        "#D55E00",  # Vermilion
           "black",  # Black
           "#56B4E9",  # Sky Blue
           "#CC79A7"]  # Reddish Purple

labels = [r'LR-$\chi^2$',
          r"Kaastra's method",
          r"Bootstrap-normal",
          r"High-order",
          r"Parametric Bootstrap"
]

markers = ["s", "o", "v", "x"]

alpha=0.8

for l in range(iters):
    plt.subplot(2, 2, 1)
    n=10
    beta=np.array([1.,1.])
    X = design_mat(beta,n,snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    s = generate_s_true(n,beta,strue,snull)
    expec=uncon_expectation(s,n,X,I)
    std = math.sqrt(uncon_var(s,n,X,I))

    C = np.zeros(B)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i]=Cashstat(x,r)

    #x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    #y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    #plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    #x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    #y = chi2fun(x, n - 2)
    #plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(0, 20, 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color=palette[0], label=labels[0],alpha=alpha,marker=markers[0],markevery=100)

    x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)),
                  0.05)
    y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    plt.plot(x, y, color=palette[1], label=labels[1],alpha=alpha,marker=markers[1],markevery=120,linestyle='dashed')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color=palette[2],linestyle='dashdot', label=labels[2],marker=markers[2],markevery=100,alpha=alpha)

    x = np.arange(expec - 3 * std, expec + 3 * std, 0.05)
    y = normfun(x, expec, std)
    plt.plot(x, y, color=palette[3], label=labels[3], marker=markers[3],markevery=100,alpha=alpha)

    #x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    #y = normfun(x, expec1, std)
    #plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True,color=palette[4],label=labels[4],alpha=alpha)
    #plt.xlabel('C-stat',fontsize=18)
    plt.ylabel('Density',fontsize=24)
    plt.xticks([0,5,10,15,20,25],fontsize=18)
    #plt.xticks([60,80,100,120,140,160],fontsize=18)
    #plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--',alpha=0.5)
    plt.legend()
    #plt.show()
    plt.title(r'Powerlaw, $n=10,K=1$', fontsize=20)

    plt.subplot(2, 2, 2)
    n=10
    beta=np.array([10.,1.])
    X = design_mat(beta,n,snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    s = generate_s_true(n,beta,strue,snull)
    expec=uncon_expectation(s,n,X,I)
    std = math.sqrt(uncon_var(s,n,X,I))

    C = np.zeros(B)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i]=Cashstat(x,r)

    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    # x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    # y = chi2fun(x, n - 2)
    # plt.plot(x, y, color='orange', label='Alg.1')

    x = np.arange(0, 20, 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color=palette[0], label=labels[0],alpha=alpha,marker=markers[0],markevery=100)

    x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)),
                  0.05)
    y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    plt.plot(x, y, color=palette[1], label=labels[1],alpha=alpha,marker=markers[1],markevery=120,linestyle='dashed')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color=palette[2],linestyle='dashdot', label=labels[2],marker=markers[2],markevery=100,alpha=alpha)

    x = np.arange(expec - 3 * std, expec + 3 * std, 0.05)
    y = normfun(x, expec, std)
    plt.plot(x, y, color=palette[3], label=labels[3], marker=markers[3],markevery=100,alpha=alpha)

    #x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    #y = normfun(x, expec1, std)
    #plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True,color=palette[4],label=labels[4], alpha=alpha)
    #plt.xlabel('C-stat', fontsize=18)
    #plt.ylabel('Density', fontsize=18)
    plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    # plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    #plt.legend()
    # plt.show()
    plt.title(r'Powerlaw, $n=10,K=10$', fontsize=20)

    plt.subplot(2, 2, 3)
    n=100
    beta=np.array([1.,1.])
    X = design_mat(beta,n,snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    s = generate_s_true(n,beta,strue,snull)
    expec=uncon_expectation(s,n,X,I)
    std = math.sqrt(uncon_var(s,n,X,I))

    C = np.zeros(B)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i]=Cashstat(x,r)

    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color=palette[0], label=labels[0],alpha=alpha,marker=markers[0],markevery=100)

    x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)),
                  0.05)
    y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    plt.plot(x, y, color=palette[1], label=labels[1],alpha=alpha,marker=markers[1],markevery=120,linestyle='dashed')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color=palette[2],linestyle='dashdot', label=labels[2],marker=markers[2],markevery=100,alpha=alpha)

    x = np.arange(expec - 3 * std, expec + 3 * std, 0.05)
    y = normfun(x, expec, std)
    plt.plot(x, y, color=palette[3], label=labels[3], marker=markers[3],markevery=100,alpha=alpha)

    #x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    #y = normfun(x, expec1, std)
    #plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True,color=palette[4],label=labels[4],alpha=alpha)
    plt.xlabel(r'$C$ statistics',fontsize=24)
    plt.ylabel('Density',fontsize=24)
    #plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    # plt.show()
    plt.title(r'Powerlaw, $n=100,K=1$', fontsize=20)

    plt.subplot(2, 2, 4)
    n=100
    beta=np.array([10.,1.])
    X = design_mat(beta,n,snull)
    I = np.asmatrix(np.repeat(1.0,len(beta))).T
    s = generate_s_true(n,beta,strue,snull)
    expec=uncon_expectation(s,n,X,I)
    std = math.sqrt(uncon_var(s,n,X,I))

    C = np.zeros(B)
    for i in range(B):
        x = poisson_data(s)
        if np.all(np.abs(x) < 1e-5):
            continue
        bound = empirical_bounds(x, snull)
        xopt = opt.minimize(LLF, beta, args=(x, snull), jac=LLF_grad, bounds=bound, method="L-BFGS-B")
        r = generate_s(n, xopt['x'], snull)
        C[i]=Cashstat(x,r)
    # x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)), 0.05)
    # y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    # plt.plot(x, y, color='green', label='Bona',linestyle='dashdot')

    x = np.arange(n - 2 - 3 * math.sqrt(2 * (n - 2)), n - 2 + 3 * math.sqrt(2 * (n - 2)), 0.05)
    y = chi2fun(x, n - 2)
    plt.plot(x, y, color=palette[0], label=labels[0],alpha=alpha,marker=markers[0],markevery=100)

    x = np.arange(Max_mean(s) - 3 * math.sqrt(Max_var(s)), Max_mean(s) + 3 * math.sqrt(Max_var(s)),
                  0.05)
    y = normfun(x, Max_mean(s), math.sqrt(Max_var(s)))
    plt.plot(x, y, color=palette[1], label=labels[1],alpha=alpha,marker=markers[1],markevery=120,linestyle='dashed')

    x = np.arange(statistics.mean(C) - 3 * statistics.stdev(C), statistics.mean(C) + 3 * statistics.stdev(C), 0.05)
    y = normfun(x, statistics.mean(C), statistics.stdev(C))
    plt.plot(x, y, color=palette[2],linestyle='dashdot', label=labels[2],marker=markers[2],markevery=100,alpha=alpha)

    x = np.arange(expec - 3 * std, expec + 3 * std, 0.05)
    y = normfun(x, expec, std)
    plt.plot(x, y, color=palette[3], label=labels[3], marker=markers[3],markevery=100,alpha=alpha)

    #x = np.arange(expec1 - 3 * std, expec1 + 3 * std, 0.05)
    #y = normfun(x, expec1, std)
    #plt.plot(x, y, color='tomato', label='Alg.3b', marker='v',markevery=100)

    plt.hist(C, bins=20, rwidth=0.9, density=True,color=palette[4],label=labels[4],alpha=alpha)
    plt.xlabel(r'$C$ statistics',fontsize=24)
    #plt.ylabel('Density', fontsize=18)
    #plt.xticks([0, 5, 10, 15, 20, 25], fontsize=18)
    plt.xticks([60,80,100,120,140,160],fontsize=18)
    # plt.yticks([0.00,0.025,0.050,0.075,0.10,0.125],fontsize=18)

    plt.grid(linestyle='--', alpha=0.5)
    # plt.show()
    plt.title(r'Powerlaw, $n=100,K=10$', fontsize=20)

    plt.tight_layout()
    plt.savefig('results/figurenew/Histogram.pdf')
