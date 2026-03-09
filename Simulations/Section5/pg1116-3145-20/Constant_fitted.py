from Simulations.Section5.Constant_fit_test import *
import matplotlib.pyplot as plt

path4test='countFormat0-segment-0-20.dat'
count=np.genfromtxt(path4test,skip_header=0)
data4test=count[:,1]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0


n=len(data4test)
B=1000
np.random.seed(42)
# constant model
snull = 'constant'
print("Constant")

x=data4test
mu_hat = np.array([np.mean(x)])
print(mu_hat)
r = generate_s_constant(n, mu_hat)

Cmin = Cashstat(x, r)
print(Cmin)

print(Simulations.utilities.Wilks_Chi2_test.p_value_chi(Cmin, n - len(mu_hat))[0])
print(Simulations.utilities.uncon_plugin.uncon_plugin_test(Cmin, mu_hat, n, snull)[0])
# print(Simulations.utilities.bootstrap_normal.bootstrap_asymptotic(Cmin,mu_hat,blockLength,snull)[0)
print(Simulations.utilities.con_theory.con_theory_test(Cmin, mu_hat, n, snull)[0])
print(Simulations.utilities.bootstrap_empirical.bootstrap_test(Cmin, mu_hat, n, snull)[0])


# Plot figure
path4figure1='countFormat0-segment-0-20.dat'
path4figure2='countFormat1-segment-0-20.dat'
count=np.genfromtxt(path4figure1,skip_header=0)
data=count[:,1]
count=np.genfromtxt(path4figure2,skip_header=0)
data2=count[:,1]
print(np.mean(data),np.mean(data2))
ytick=[0,np.mean(data),1,np.mean(data2),2,3,4,5,6]
ylabel=['0','0.429','1','1.497','2','3','4','5','6']
wavelength=[20.0+.0125/2+.0125*i for i in range(n)]
xerror=0.0125
fig,ax=plt.subplots(figsize=(8,6))
#plt.errorbar(wavelength,data,xerr=xerror,yerr=data**0.5,alpha=0.3,color='r',label='NEW OBS SEG1')
plt.scatter(wavelength,data,color='r',label='2018 OBS SEG1',s=6,marker='.',alpha=0.8)
#plt.errorbar(wavelength,data2,xerr=xerror,yerr=data2**0.5,alpha=0.3,color='b',label='BACK NEW OBS SEG1')
plt.scatter(wavelength,data2,color='b',label='BACK 2018 OBS SEG1',s=6,marker='d',alpha=0.8)
plt.axhline(np.mean(data),alpha=0.8,color='r',linestyle='-')
plt.axhline(np.mean(data2),alpha=0.8,color='b',linestyle='-')
plt.grid(alpha=0.5)
plt.xlabel('Wavelength (Ang.)')
plt.ylabel('Counts')
plt.ylim(-0.1,6.2)
plt.xlim((wavelength[0]-xerror,wavelength[n-1]+xerror))
plt.legend(fontsize=15)
plt.yticks(ytick,ylabel)
plt.savefig('pg1116CountSpectrumSegment1.pdf')
