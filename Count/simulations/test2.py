from copy import deepcopy

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import ortho_group

from matplotlib import pyplot as plt, scale as mscale, colors, cm, lines as mlines

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, zero_one_loss
from sklearn.datasets import make_low_rank_matrix

from tqdm import tqdm

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
plt.rcParams['figure.figsize'] = [6, 2.2]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 'small'
plt.rcParams['axes.titlepad'] = 3
plt.rcParams['xtick.labelsize'] = 'x-small'
plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.handlelength'] = 1.2
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.edgecolor'] = '#333'
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.9
plt.rcParams['patch.linewidth'] = 0.9
plt.rcParams['hatch.linewidth'] = 0.9
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.major.width'] = plt.rcParams['xtick.major.width']
plt.rcParams['ytick.minor.width'] = plt.rcParams['xtick.minor.width']

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def make_low_rank_spd_matrix(n, rank):
    A = make_low_rank_matrix(n, n, effective_rank=rank)
    A = A * A.T
    return A


def generate_dist_shift(d, kP, kQ, sigma_beta=1, kappa_gamma=1, rho=1):
    if kP + kQ > 1:
        raise ValueError('cannot have kP + kQ > 1')

    dP = int(kP * d)
    dQ = int(kQ * d)

    SigmaP = np.ones(d) / d
    SigmaP[-dQ:] = 0

    beta = np.random.randn(d) * sigma_beta

    p = (1 / kappa_gamma - 1) / 2
    s = np.abs(beta) ** p * sigma_beta / np.sqrt(np.mean(np.abs(beta) ** (2 * (p + 1))))

    SigmaQ = rho * s ** 2 / d
    SigmaQ[:dP] = 0

    return SigmaP, SigmaQ, beta


def generate_dist_shift_noncommute(d, kP, kQ, sigma_beta=1, kappa_gamma=1, rho=1):
    if kP + kQ > 1:
        raise ValueError('cannot have kP + kQ > 1')

    dQ = d-int(kP * d)
    dP = d-int(kQ * d)

    Ortho=np.float32(ortho_group.rvs(size=1, dim=d))
    UP=Ortho[:,0:dP]

    beta =np.random.randn(d) * sigma_beta

    Ortho=np.float32(ortho_group.rvs(size=1, dim=d))
    UQ=Ortho[:,0:dQ]

    return UP, UQ, beta


def gen_train_test_noncommute(UP, UQ, n_train, n_test, dPR):
    d=1
    dp = UP.shape[1]
    dq = UQ.shape[1]

    X_train = np.random.randn(n_train, dp)/d @ UP.T

    X_P = np.random.randn(n_test, dp)/d @ UP.T
    X_Q = np.random.randn(n_test, dq)/d @ UQ.T

    return X_train, X_P, X_Q


def gen_train_test(SigmaP, SigmaQ, n_train, n_test, dPR):
    d = len(SigmaP)

    X_train = np.random.randn(n_train, dPR) * np.sqrt(SigmaP[:dPR])[None, :]

    X_P = np.random.randn(n_test, d) * np.sqrt(SigmaP)[None, :]
    X_Q = np.random.randn(n_test, d) * np.sqrt(SigmaQ)[None, :]

    return X_train, X_P, X_Q


def gen_additive_labels(*Xs, beta=None, sigma=1):
    if beta is None:
        raise ValueError('must supply beta')

    ys = []

    for X in Xs:
        n, d = X.shape
        # X may be only in the first subspace for training data, so clip if necessary
        beta_clip = beta[:d]
        y = X @ beta_clip + np.random.randn(n) * sigma
        ys.append(y)

    if len(ys) == 1:
        return ys[0]
    else:
        return ys


def gen_binary_labels(*Xs, beta=None, p_err=0):
    if beta is None:
        raise ValueError('must supply beta')

    ys = []

    for X in Xs:
        n, d = X.shape
        # X may be only in the first subspace for training data, so clip if necessary
        beta_clip = beta[:d]
        y = (X @ beta_clip > 0).astype(float)
        to_flip = np.random.rand(n) < p_err
        y[to_flip] = 1 - y[to_flip]
        ys.append(y)

    if len(ys) == 1:
        return ys[0]
    else:
        return ys


def train_ridge(X_train, y_train, alphas):
    ridge = Ridge(alpha=alphas[0], fit_intercept=False)

    models = []

    for alpha in tqdm(alphas):
        ridge.set_params(alpha=alpha)
        models.append(deepcopy(ridge.fit(X_train, y_train)))

    return models


def train_logistic_ridge(X_train, y_train, Cs):
    ridge = LogisticRegression(solver='lbfgs', penalty='l2', C=Cs[0], warm_start=True, fit_intercept=False)

    models = []

    for C in tqdm(Cs):
        ridge.set_params(C=C)
        models.append(deepcopy(ridge.fit(X_train, y_train)))

    return models


def train_to_full_coef(model, d):
    coef_ = model.coef_.ravel()
    dPR = len(coef_)

    coef = np.zeros(d)
    coef[:dPR] = coef_.copy()

    return coef


def eval_risks(*Xys, beta=None, metric=mean_squared_error):
    if beta is None:
        raise ValueError('must supply beta')

    risks = []

    for X, y in Xys:
        if metric is zero_one_loss:
            y_hat = (X @ beta > 0).astype(float)
        else:
            y_hat = X @ beta
        risk = metric(y, y_hat)
        risks.append(risk)

    return risks


def eval_disagreements(*Xys, beta1=None, beta2=None, metric=mean_squared_error):
    if beta1 is None or beta2 is None:
        raise ValueError('must supply beta')

    risks = []

    for X, y in Xys:
        if metric is zero_one_loss:
            y1_hat = (X @ beta1 > 0).astype(float)
            y2_hat = (X @ beta2 > 0).astype(float)
        else:
            y1_hat = X @ beta1
            y2_hat = X @ beta2
        risk = metric(y1_hat, y2_hat)
        risks.append(risk)

    return risks

def theory_risks(alphas,beta,SigmaP,SigmaQ):
    d=SigmaP.shape[0]
    RPs=[]
    RQs=[]
    temp=np.array([1/(1+regpara) for regpara in alphas])
    for regpara in alphas:
        alpha=1/(1+regpara)
        c=(beta.T @ SigmaP @ beta)[0,0]
        #print(alpha,c)
        RP=(1+alpha**2-2*alpha)*c+sigma**2
        RPs.append(RP)
        c=(beta.T@SigmaP.T@SigmaQ@SigmaP@beta)[0,0]
        e=(beta.T@SigmaP.T@SigmaQ@(np.eye(d)-SigmaP)@beta)[0,0]
        print((beta.T@SigmaQ@beta)[0,0],c,e)
        RQ=(beta.T@SigmaQ@beta)[0,0]+(alpha**2-2*alpha)*c-2*alpha*e+sigma**2
        RQs.append(RQ)
    #print(temp)
    #print(RPs)
    #print(RQs)
    return RPs, RQs

d_n = 0.8
n_test = 1000
n_train = 1000
n_trials = 20
kP = 0.45
kQ = 0.45
kR = 1 - kP - kQ
sigma_beta = 1
kappa_gamma = 1
rho = 2
alphas = np.logspace(-3, 6)
#alphas=np.array([0.0001*(i+1) for i in range(5000)])
sigma = np.sqrt(0.2)

#np.random.seed(142)  #nonlinear but monontone
np.random.seed(192)  # nonlinear and not monotone

#d = int(n_train * d_n)
d=30
dQ = int(kQ * d)
dPR = d - dQ
risks_P = np.zeros((n_trials, len(alphas)))
risks_Q = np.zeros_like(risks_P)

disagreements_P = np.zeros((n_trials, len(alphas)-1))
disagreements_Q = np.zeros_like(disagreements_P)

UP, UQ, beta = generate_dist_shift_noncommute(d, kP, kQ, sigma_beta=sigma_beta, kappa_gamma=kappa_gamma,
                                                          rho=rho)
betastar = np.mat(beta).T
SigmaP = UP @ UP.T
SigmaQ = UQ @ UQ.T

for t in range(n_trials):
    #SigmaP, SigmaQ, beta = generate_dist_shift_noncommute(d, kP, kQ, sigma_beta=sigma_beta, kappa_gamma=kappa_gamma, rho=rho)
    #beta = np.random.randn(d)*sigma_beta
    betastar=np.mat(beta).T
    SigmaP=UP@UP.T
    SigmaQ=UQ@UQ.T
    print(betastar.T@SigmaP@SigmaQ@(np.eye(d)-SigmaP)*betastar)
    print((betastar.T@SigmaP@SigmaQ@SigmaP*betastar)[0,0]/(betastar.T@SigmaP@betastar)[0,0])
    X_train, X_P, X_Q = gen_train_test_noncommute(UP, UQ, n_train, n_test, dPR)
    y_train = gen_additive_labels(X_train, beta=beta, sigma=sigma)
    y_P, y_Q = gen_additive_labels(X_P, X_Q, beta=beta, sigma=0)
    ridges = train_ridge(X_train, y_train, alphas)

    betas = []

    for j, ridge in enumerate(ridges):
        beta_hat = train_to_full_coef(ridge, d)
        betas.append(beta_hat)
        risk_P, risk_Q = eval_risks((X_P, y_P), (X_Q, y_Q), beta=beta_hat)
        risks_P[t, j] = risk_P
        risks_Q[t, j] = risk_Q

    l = 0
    for j in range(len(alphas)-1):
        disagreement_P, disagreement_Q = eval_disagreements((X_P, y_P), (X_Q, y_Q), beta1=betas[j], beta2=betas[len(alphas)-1])
        disagreements_P[t, l] = disagreement_P
        disagreements_Q[t, l] = disagreement_Q
        l = l + 1

rP = kP + kR
gamma = kR / rP * rho
kappa = gamma * kappa_gamma
mu = (kQ + kR) / kR

mse_offset = gamma * rP * (mu - 1) * sigma_beta ** 2
plt.subplot(1,2,1)
for t in range(n_trials):
    plt.plot(disagreements_P[t, :], disagreements_Q[t, :], color=color_cycle[0], alpha=0.2)

R_P, R_Q = disagreements_P.mean(0), disagreements_Q.mean(0)
std_P, std_Q = disagreements_P.std(0), disagreements_Q.std(0)


plt.scatter(R_P, R_Q)
#plt.errorbar(R_P, R_Q, yerr=std_Q, xerr=std_P, fmt='o')

slope_agree=(betastar.T@SigmaP@SigmaQ@SigmaP*betastar)[0,0]/(betastar.T@SigmaP@betastar)[0,0]

plt.xlabel('D_P')
plt.ylabel('D_Q')
plt.title("Disagreement of Regression")
plt.plot(R_P, slope_agree * R_P , linestyle='--',color='blue')

#plt.plot(R_P, gamma * R_P + mse_offset, '--k')
#plt.show()

rP = kP + kR
gamma = kR / rP * rho
kappa = gamma * kappa_gamma
mu = (kQ + kR) / kR

mse_offset = gamma * rP * (mu - 1) * sigma_beta ** 2
plt.subplot(1,2,2)
for t in range(n_trials):
    plt.plot(risks_P[t, :], risks_Q[t, :], color=color_cycle[0], alpha=0.2)

R_P, R_Q = risks_P.mean(0), risks_Q.mean(0)
std_P, std_Q = risks_P.std(0), risks_Q.std(0)

plt.scatter(R_P, R_Q)
#plt.errorbar(R_P, R_Q, yerr=std_Q, xerr=std_P, fmt='o')

beta=np.mat(beta).T
RPs,RQs=theory_risks(alphas, beta, SigmaP, SigmaQ)

plt.xlabel('R_P')
plt.ylabel('R_Q')
plt.title("Risks of Regression")
plt.plot(RPs, RQs,linestyle= '--',color='red')
plt.tight_layout()
plt.savefig('regression.png')

#plt.plot(R_P, gamma * R_P + mse_offset, '--k')
#plt.show()