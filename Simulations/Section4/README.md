
# Section 4 – Simulation Studies

This folder contains the code, data, and results for reproducing **Section 4** of our paper.

---

## 📄 Overview
In Section 4, we evaluate the performance of our proposed method under various settings.
The main conclusion is that our recommended Corrected $Z$-test is best calibrated in terms of Type I error rate and power.

---

## ▶️ How to Reproduce
1. Make sure you have installed the dependencies from the main README.
   
2. Run:
bash run.sh

This will:
- Load the data from `data/`
- Run the main script in `code/`
- Save the figures to `results/`

## 📊 Results
Example output (Figure 1 from the paper):

## Simulation Settings

In our simulation design, we set $\Gamma= 1$, and treat $n\in \{10, 25, 50, 100, 200 \}$, and $K\in \{0.1, 0.25, 0.5, 1, 1.6, 2.5, 5,10 \}$ as factors. The entire set of simulations is repeated with (a) no line, i.e., $b=0$, (b) an emission line with $\Phi = 2K$, and (c) an absorption line with $\Psi = K/10$. In both (b) and (c) the line is located with $m_1=n/2$ and $b=n/10$ so that the line extends over the energy range $(1.5, 1.6)$, but shifted one-half of a bin width to the right.

You can reproduce a specific setting by:

Change the parameters in the code `Cstat_test_with_single.py`. The default setting is 

`
    # params

    n = 500  # number of bins
    
    B = 300
    
    beta = np.array([0.25, 1])  # ground-truth beta*
    
    strue = 'powerlaw'  # true s : powerlaw/ brokenpowerlaw/ spectral_line
    
    snull = 'powerlaw'  # s of H_0 : powerlaw
    
    loc, strength, width = [0.5, 3, int(0.1*n)]  # For broken-powerlaw and spectral line
    
    iters = 3000  # repetition times, suppose p=0.1. Then CI = +-0.01 (3k), +-0.02 (1k). p=0.25, then CI = +-0.015 (3k). p=0.5, CI = +-0.02( 3k)
    
    np.random.seed(0)  # random seed
    `


Then
```bash
python Cstat_test_with_single.py
