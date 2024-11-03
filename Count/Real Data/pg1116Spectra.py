import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

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
print(countSpectrum[0])

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
        Model[i][j]=countSpectrum[i][j][4]*DeltaLambda[i][j]
        print('%.3f %d (%f) %.3f\n'%(Lambda[i][j],Counts[i][j],countSpectrum[i][j][2]*DeltaLambda[i][j],Model[i][j]))
        fp.write('%.3f %d %.3f\n'%(Lambda[i][j],Counts[i][j],Model[i][j]))
    # Now check the cmin statistics for the 4 datasets
    cminSpectrum[i]=cstat(Counts[i],Model[i])
    print('Spectrum %d has cmin=%.3f for %d bins'%(i,sum(cminSpectrum[i]),len(Counts[i])))

# Also plot them
# (a) Plot the counts
fig,ax=plt.subplots(figsize=(8,6))
i=0
plt.errorbar(Lambda[i],Counts[i],xerr=DeltaLambda[i],yerr=Counts[i]**0.5,ls='',color='black',label='OLD OBS.')
plt.plot(Lambda[i],Model[i],color='black',ls='-')
i=1
plt.errorbar(Lambda[i],Counts[i],xerr=DeltaLambda[i],yerr=Counts[i]**0.5,ls='',color='red',label='NEW OBS.')
plt.plot(Lambda[i],Model[i],color='red',ls='-')
i=3
plt.errorbar(Lambda[i],Counts[i],xerr=DeltaLambda[i],yerr=Counts[i]**0.5,ls='',color='grey',label='BACK OLD OBS.')
plt.plot(Lambda[i],Model[i],color='grey',ls='-')
i=2
plt.errorbar(Lambda[i],Counts[i],xerr=DeltaLambda[i],yerr=Counts[i]**0.5,ls='',color='orange',label='BACK NEW OBS.')
plt.plot(Lambda[i],Model[i],color='orange',ls='-')
plt.legend(fontsize=12)
plt.grid()
plt.xlabel('Wavelength (Ang.)')
plt.ylabel('Counts')
plt.xlim((min(Lambda[0]-DeltaLambda[0]),max(Lambda[0]+DeltaLambda[0])))
plt.savefig('pg1116CountSpectrum.pdf')      

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


