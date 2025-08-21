from HEAGOF.utilities.utilities import *
import HEAGOF.utilities.Wilks_Chi2_test, HEAGOF.utilities.uncon_plugin, HEAGOF.utilities.con_theory
import HEAGOF.utilities.bootstrap_empirical, HEAGOF.utilities.bootstrap_double


class model_class:
    def __init__(self,label='powerlaw',func=powerlaw_func,initializer=np.ones(2),bound=None):
        self.label=label
        self.func=func
        self.initializer=initializer
        self.bound=bound

class HEAGOF_class:

    def __init__(self, data=np.array([]), model=model_class(), energy=np.array([0]), effective_area=np.array([]), redistribution_matrix=np.eye(0), back_strength=0.):
        '''
        Initialize the HEAGOF object:
            data: observed counts, 1D array of counts, length is Blocklength

            model: model used to fit the data, a class with label and func
                example (powerlaw):
                        class powerlaw_model:
                            def powerlaw_func(self,x,theta):
                                return theta[0] * x ** (-theta[1])

                            def __init__(self):
                                self.label='powerlaw_model'
                                self.func=self.powerlaw_func

            energy: energy range, 1D array of increasing positive numbers, length is n_total+1

            effective_area: Effective area range, 1D array of positive numbers, length is n_total

            redistribution_matrix: Redistribution matrix, 2D matrix of positive numbers
                with shape Blocklength x n_total

            back_strength: Strength of the background, positive float number

        '''
        Blocklength, n_total = redistribution_matrix.shape
        if len(data)!=Blocklength:
            raise ValueError('The shape of the redistribution matrix does not match the data!')
        #if n_total!=Blocklength:
        #    raise ValueError('The redistribution matrix is not a square matrix!')
        if len(energy)!=(Blocklength+1):
            raise ValueError('The shape of the energy range must be n_total+1 !')
        self.data = data

        self.model_label = model.label
        self.model_func = model.func
        self.initializer = model.initializer
        self.bound=model.bound

        self.energy_band = energy[1:]-energy[:Blocklength] #Energy band width
        self.energy_mid = (energy[:Blocklength]+energy[1:])/2
        self.RMF=redistribution_matrix
        self.ARE=effective_area
        self.back_strength = back_strength

        self.thetahat = None
        self.cashstat = None

    def generate_s(self,theta):
        '''
        Generate Poisson rates
        '''
        s_without_rmf=np.array([self.model_func(x,theta) for x in self.energy_mid])
        return self.RMF@(self.ARE*s_without_rmf*self.energy_band)+self.back_strength

    def LLF(self, theta, x):
        '''
        Likelihood function
        '''
        eps = 1e-12
        s = self.generate_s(theta)
        return -np.sum(x * np.log(s + eps) - s)

    def fit(self):
        '''
        Fit the model to get the maximum likelihood estimator
        '''
        print("Fitting the model...")
        if np.all(np.abs(self.data) < 1e-5):
            raise ValueError("No counts observed!")
        xopt = opt.minimize(self.LLF, self.initializer,args=(self.data),
                            bounds=self.bound)['x']
        self.thetahat = xopt
        print(f"Fitted value is {self.thetahat}")

    def cashstat_calculate(self):
        '''
        Calculate Cash statistic
        '''
        if self.thetahat is None:
            self.fit()
        self.s_fitted=self.generate_s(self.thetahat)
        self.cashstat=Cashstat(self.data,self.s_fitted)
        print(f"Cash statistic: {self.cashstat}")

    def design_matrix(self):
        '''
        Generate design matrix
        '''
        if self.thetahat is None:
            self.fit()
        self.fitted_design_matrix=numerical_jacobian(self.generate_s, self.thetahat)



def goodness_of_fit(model=HEAGOF_class(),method='theory'):
    if model.cashstat is None:
        raise ValueError("Please run .cashstat_calculate() first!")
    if method=='chi2':
        model.pvalue=HEAGOF.utilities.Wilks_Chi2_test.p_value_chi(model)[0]
    elif method=='plug-in':
        model.pvalue=HEAGOF.utilities.uncon_plugin.uncon_plugin_test(model)[0]
    elif method=='theory':
        model.pvalue=HEAGOF.utilities.con_theory.con_theory_test(model)[0]
    elif method=='bootstrap':
        model.pvalue=HEAGOF.utilities.bootstrap_empirical.bootstrap_test(model, Cmin=model.cashstat, thetahat=model.thetahat,B=1000)[0]
    elif method=='bootstrap_double':
        model.pvalue=HEAGOF.utilities.bootstrap_double.double_boostrap(model,B1=1000,B2=1000)[0]
    else:
        raise ValueError("Method must be either 'chi2', 'theory', 'plug-in', 'bootstrap' or 'bootstrap_double'")

    print(f"The p-value is: {model.pvalue}")


if __name__=="__main__":
    # Example: powerlaw model

    eps = 1e-5
    mylabel = 'powerlaw'
    myfunc = powerlaw_func
    myinitializer = np.ones(2)
    mybound = [[eps, 1 / eps], [-2 * math.log((10 + 2 * eps) / (eps), 2),
                                   2 * math.log((10 + 2 * eps) / (eps), 2)]]
    mymodel=model_class(mylabel,myfunc,myinitializer,mybound)

    # Example: test a powerlaw model for a real powerlaw model when RMF is identity
    myn=1000
    mytheta=np.array([0.1, 1])
    myenergy=np.linspace(1,2,myn+1)
    myARE=np.ones(myn)
    myRMF=np.eye(myn)
    np.random.seed(0)

    mys=generate_s(myn,mytheta,snull='powerlaw')
    mydata=poisson_data(mys)

    myclass=HEAGOF_class(data=mydata,model=mymodel,energy=myenergy,effective_area=myARE,redistribution_matrix=myRMF,back_strength=0.)
    myclass.fit()
    myclass.cashstat_calculate()
    myclass.design_matrix()

    # p-values given by 4 algorithm:
    goodness_of_fit(model=myclass,method='chi2')
    goodness_of_fit(model=myclass, method='plug-in')
    goodness_of_fit(model=myclass, method='theory')
    goodness_of_fit(model=myclass, method='bootstrap')