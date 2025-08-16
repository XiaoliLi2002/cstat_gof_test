from HEPGOF.utilities import *
import HEPGOF.Wilks_Chi2_test
import HEPGOF.uncon_plugin,  HEPGOF.con_theory, HEPGOF.bootstrap_empirical, HEPGOF.bootstrap_normal


if __name__=="__main__":
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

    cminSpectrum=np.zeros((nBlocks,blockLength))
    for i in range(nBlocks):
        fp=open('countFormat%d.dat'%i,'w')
        for j in range(blockLength):
            Lambda[i][j]=countSpectrum[i][j][0]
            DeltaLambda[i][j]=countSpectrum[i][j][1]*2 # Delta Lambda
        # Need to round off to nearest integer
            Counts[i][j]=round(countSpectrum[i][j][2]*DeltaLambda[i][j])

    B=1000
    np.random.seed(42)
    for i in range(nBlocks):
        x=Counts[i]

    #constant model
        snull='constant'
        print("Constant")

        mu_hat=np.array([np.mean(x)])
        print(mu_hat)
        r = generate_s_constant(blockLength, mu_hat)

        Cmin=Cashstat(x,r)
        print(Cmin)

        print(HEPGOF.Wilks_Chi2_test.p_value_chi(Cmin,blockLength-len(mu_hat)))
        print(HEPGOF.uncon_plugin.uncon_plugin_test(Cmin,mu_hat,blockLength,snull))
        #print(HEPGOF.bootstrap_normal.bootstrap_asymptotic(Cmin,mu_hat,blockLength,snull))
        print(HEPGOF.con_theory.con_theory_test(Cmin,mu_hat,blockLength,snull))
        print(HEPGOF.bootstrap_empirical.bootstrap_test(Cmin,mu_hat,blockLength,snull))

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

