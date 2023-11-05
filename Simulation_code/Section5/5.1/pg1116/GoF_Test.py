import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import scipy.optimize as opt
import scipy
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

def normfun(x,mu,sigma):
    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf

def generate_s_exp(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*math.exp(-theta2*i/n)
    return arr

def generate_s_constant(n,mu):
    return [mu for i in range(n)]

def generate_s_powerlaw(n,theta1,theta2):
    arr=[0 for x in range(n)]
    for i in range(n):
        arr[i]=theta1*((1+(i+1)/n)**theta2)
    return arr

def poisson_data(mu):
    arr=[0 for x in range(len(mu))]
    for i in range(len(mu)):
        arr[i]=poisson.rvs(mu=mu[i],size=1)[0]
    return arr

def LLF_exp(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*(math.log(theta[0],math.e)-theta[1]*(i+1)/n)-theta[0]*math.e**(-theta[1]*(i+1)/n)
    return -value

def LLF_powerlaw(theta):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*math.log(theta[0]*((1+(i+1)/n)**theta[1]),math.e)-theta[0]*((1+(i+1)/n)**theta[1])
    return -value

def LLF_constant(mu):
    n=len(x)
    value=0
    for i in range(n):
        value+=x[i]*math.log(mu[0],math.e)-mu[0]
    return -value

def p_value_norm(x,mu,sigma):
    z=(x-mu)/sigma
    return scipy.stats.norm.sf(z)

def p_value_chi(x,df):
    return scipy.stats.chi2.sf(x,df)

max=300
def poisson_dis(mu,i):
    return poisson.pmf(i,mu)

def kapa_1(mu):
    x=0
    for k in range(max):
        if k==0:
            x-=2*(k-mu)*poisson_dis(mu,k)
        else:
            x+=2*(k*math.log(k/mu,math.e)-k+mu)*poisson_dis(mu,k)
    return x

def kapa_2(mu):
    k_1=kapa_1(mu)
    x=0
    for k in range(max):
        if k==0:
            x+=(-2*(k-mu)-k_1)**2*poisson_dis(mu,k)
        else:
            x+=(2*(k*math.log(k/mu,math.e)-k+mu)-k_1)**2*poisson_dis(mu,k)

    return x

def kapa_11(mu):
    k_1=kapa_1(mu)
    x=0
    for k in range(max):
        if k==0:
            x+=(-2*(k-mu)-k_1)*(k-mu)*poisson_dis(mu,k)
        else:
            x+=(2*(k*math.log(k/mu,math.e)-k+mu)-k_1)*(k-mu)*poisson_dis(mu,k)
    return x

def kapa_12(mu):
    k_1 = kapa_1(mu)
    x = 0
    for k in range(max):
        if k == 0:
            x += (-2 * (k - mu) - k_1) * (k - mu)**2 * poisson_dis(mu, k)
        else:
            x += (2 * (k * math.log(k / mu, math.e) - k + mu) - k_1) * (k - mu)**2 * poisson_dis(mu, k)
    return x

def kapa_03(mu):
    return mu

def Sigma_diag(s,i,Q,n):
    x=kapa_12(s[i])
    for j in range(n):
        x-=kapa_11(s[j])*Q[j,i]*kapa_03(s[i])
    return x

def expectation(s,n):
    V = np.diag([s[i] for i in range(n)])
    Q=X*(X.T*V*X)**(-1)*X.T
    Sigma=np.diag([Sigma_diag(s,i,Q,n) for i in range(n)])

    E=(-0.5*I.T*X.T*Sigma*X*(X.T*V*X)**(-1)*I)[0]
    for i in range(n):
        E+=kapa_1(s[i])
    print(float(E))
    return float(E)

def Var(s,n):
    V = np.diag([s[i] for i in range(n)])
    k_11=np.mat([kapa_11(s[i]) for i in range(n)]).T
    var=(-k_11.T*X*(X.T*V*X)**(-1)*X.T*k_11)[0]
    for i in range(n):
        var+=kapa_2(s[i])
    print(math.sqrt(var))
    return var

def theory_test(x,s,n):
    return p_value_norm(x,expectation(s,n),math.sqrt(Var(s,n)))

# Read in the data in XSPEC format
count=np.genfromtxt('count.dat',skip_header=3)
print(count)

# Now read individual blocks (one for each spectrum)
blockLength=159
nBlocks=4
nCols=5 # 5 columns for the XSPEC spectrum
countSpectrum=np.zeros((nBlocks,blockLength,nCols))
iStart=0
iEnd=blockLength
for i in range(nBlocks):
    print('doing block %d'%i)
    countSpectrum[i,:]=count[iStart:iEnd,:] #*blockLength:(i+1)*blockLength]
    # update the counters
    iStart=iStart+blockLength+1 # +1 because of the NO NO separator line
    iEnd=iEnd+blockLength+1

# 0: OLD DATA
# 1: NEW DATA
# 2: NEW BACK
# 3: OLD BACK
# Now reformat the spectrum in counts/bin; they are in c/Angstrom
Lambda=np.zeros((nBlocks,blockLength))
DeltaLambda=np.zeros((nBlocks,blockLength))
Counts=np.zeros((nBlocks,blockLength),dtype=int)
Model=np.zeros((nBlocks,blockLength),dtype=float)
# Also check that the XSPEC cmin statistics are OK
from StatFunctions import cstat
cminSpectrum=np.zeros((nBlocks,blockLength))
for i in range(nBlocks):
    fp=open('countFormat%d.dat'%i,'w')
    for j in range(blockLength):
        Lambda[i][j]=countSpectrum[i][j][0]
        DeltaLambda[i][j]=countSpectrum[i][j][1]*2 # Delta Lambda
        # Need to round off to nearest integer
        Counts[i][j]=round(countSpectrum[i][j][2]*DeltaLambda[i][j])

B=1000
for i in range(nBlocks):
    x=Counts[i]

    #constant model
    print("Constant")
    X = np.mat([[1 for x in range(blockLength)]]).T
    I = np.mat([1.]).T
    mu_hat=np.mean(x)
    print(mu_hat)
    r = generate_s_constant(blockLength, mu_hat)
    Cmin = 0
    for j in range(blockLength):
        if x[j] == 0:
            Cmin += 2 * r[j]
        else:
            Cmin += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
    print(Cmin)
    s = generate_s_constant(blockLength, mu_hat)
    C = [0 for x in range(B)]
    for i in range(B):
        x = poisson_data(s)
        mu_mle=np.mean(x)
        r = generate_s_constant(blockLength, mu_hat)
        for j in range(blockLength):
            if x[j] == 0:
                C[i] += 2 * r[j]
            else:
                C[i] += 2 * (r[j] - x[j] * math.log(r[j], math.e) - x[j] + x[j] * math.log(x[j], math.e))
    print(statistics.mean(C), statistics.stdev(C), p_value_norm(Cmin, statistics.mean(C), statistics.stdev(C)))
    tem = 0
    C_re = sorted(C)
    for i in range(B):
        if C_re[i] < Cmin:
            i += 1
            tem += 1
        else:
            print((B - i) / B)
            break
    print(p_value_chi(Cmin, blockLength - 1))
    print(theory_test(Cmin, s, blockLength))

# (b) Since the exposure times of observations are different, normalize by exposure times
EXPTIMES=[8.805e+04,2.674e+05]
# Also, BACK spectra differ from SOURCE spectra of same observation, in two ways
# 1. they come from a larger area, as measured by the BACKSCAL parameter
BACKSCAL=[10.0,10.0]
# 2. they have different "effective area" (BACK does not go through the optics)
EFFAREARATIO=[16.565,13.968]
# These are all deterministic values; BASCKSCAL and EFFAREARATIO are already accounted for
# in the XSPEC analysis, by renormalizing the best-fit BACK model when applied to the
# SOURCE spectrum

