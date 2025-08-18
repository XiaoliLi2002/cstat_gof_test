
# High-Energy Astronomy Goodness-of-Fit (HEAGOF) - a mini package for appropriately doing goodness-of-fit test for astronomical count data

This dictory contains the package $\texttt{HEAGOF}$

---

## 📂 Structure
- `HEAGOF/utilities/` – Helper functions used in `main.py`
- `main.py` - main script

---

## ⚙️ Analysis of `main.py`

In this package, we define two classes `model_class`  and `HEAGOF_class` :

    class model_class:
        def __init__(self,label='powerlaw',func=powerlaw_func,initializer=np.ones(2),bound=None):
            self.label=label
            self.func=func
            self.initializer=initializer
            self.bound=bound

`model_class` contains attributes:

- `.label`: label of this model, a string
- `.func`: function $g(\tilde{E}_j, \boldsymbol{\theta})$ in our paper Eq.(5), which charaterize the model
- `.initializer`: initializer for optimization
- `.bound`: bound of parameters

    class HEAGOF_class:

        def __init__(self, data=np.array([]), model=powerlaw_model(), energy=np.array([0]), effective_area=np.array([]), redistribution_matrix=np.eye(0), back_strength=0.):
            '''
            Initialize the HEAGOF object:
                data: observed counts, 1D array of counts, length is Blocklength

                model: model used to fit the data, a class with label and func
                    example (powerlaw):
                            class constant_model: # Example: constant model
                                def constant_func(self,x,mu):
                                    return mu[0]

                                def __init__(self):
                                    eps = 1e-5
                                    self.label='constant'
                                    self.func=self.constant_func
                                    self.initializer=np.ones(1)
                                    self.bound=[[eps, 1/eps]]

                energy: energy range, 1D array of increasing positive numbers, length is Blocklength+1

                effective_area: Effective area range, 1D array of positive numbers, length is Blocklength

                redistribution_matrix: Redistribution matrix, 2D matrix of positive numbers
                with shape Blocklength x Blocklength

                back_strength: Strength of the background, positive float number

            '''
`HEAGOF_class` contains attributes:

- `.generate_s(self,theta)`: generate Poisson rates based on entered model
- `.LLF(self, theta, x)`: calculate the value of likelihood function
- `.fit(self)`: fit entered model to get maximum likelihood estimator
- `.cashstat_calculate(self)`: Calculate Cash statistic
- `.design_matrix(self)`: Generate design matrix

To test a specified spectral model for given data, you need:

- Define a `model_class`, which contains information about the model and parameter space.
- Define a `HEAGOF_class`, which makes the model practical with given data and information about the observer, including the energy range, redistribution matrix, effective area and strength from the background.
- Run `.fit()`, `.cashstat_calculate()` and `.design_matrix()` to fit the model and get necessary quantities.
- Run goodness_of_fit with the model and a method chosen from `'chi2'`, `'plug-in'`, `'theory'`, `'bootstrap'` and `'bootstrap_double'`.
- Then this function will print the $p$-value and save it as an attribute `.pvalue`.

---

## 🚀 Quick Start
Example for using our package to test a Powerlaw model:

`cd HEAGOF`

run `main.py`

This will:

- Created a synthetic data based on a powerlaw spectrum with parameters entered
- Define a powerlaw model with `model_class`
- Define a practical powerlaw model based on synthetic data and information about the observer with `HEAGOF_class`
- Fit the model, calculate the Cash statistic and do goodness of fit test with methods `'chi2'`, `'plug-in'`, `'theory'` and `'bootstrap'`.



