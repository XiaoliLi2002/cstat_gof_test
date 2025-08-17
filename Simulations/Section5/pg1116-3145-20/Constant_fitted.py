from Simulations.Section5.Constant_fit_test import *

path='countFormat0-segment-0-20.dat'
path4test='countFormat0-segment-0-20.dat'
count=np.genfromtxt(path,skip_header=0)
data=count[:,1]
count=np.genfromtxt(path4test,skip_header=0)
data4test=count[:,1]


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

print(Simulations.utilities.Wilks_Chi2_test.p_value_chi(Cmin, n - len(mu_hat)))
print(Simulations.utilities.uncon_plugin.uncon_plugin_test(Cmin, mu_hat, n, snull))
# print(Simulations.utilities.bootstrap_normal.bootstrap_asymptotic(Cmin,mu_hat,blockLength,snull))
print(Simulations.utilities.con_theory.con_theory_test(Cmin, mu_hat, n, snull))
print(Simulations.utilities.bootstrap_empirical.bootstrap_test(Cmin, mu_hat, n, snull))


# Plot figure
#ytick=[0,1,np.mean(data),2,np.mean(data2),3,4,5]
#wavelength=[20.0+.0125/2+.0125*i for i in range(n)]
#xerror=0.0125
#fig,ax=plt.subplots(figsize=(8,6))
#plt.errorbar(wavelength,data,xerr=xerror,yerr=data**0.5,alpha=0.3,color='r',label='NEW OBS SEG1')
#plt.scatter(wavelength,data,color='r',label='NEW OBS SEG1',s=6,marker='+')
#plt.errorbar(wavelength,data2,xerr=xerror,yerr=data2**0.5,alpha=0.3,color='b',label='BACK NEW OBS SEG1')
#plt.scatter(wavelength,data2,color='b',label='BACK NEW OBS SEG1',s=6,marker='+')
#plt.axhline(np.mean(data),alpha=1,color='r',linestyle='-')
#plt.axhline(np.mean(data2),alpha=1,color='b',linestyle='-')
#plt.grid(alpha=0.5)
#plt.xlabel('Wavelength (Ang.)')
#plt.ylabel('Counts')
#plt.xlim((wavelength[0]-xerror,wavelength[n-1]+xerror))
#plt.legend(fontsize=12)
#plt.yticks(ytick)
#plt.savefig('pg1116CountSpectrumSegment1.pdf')
