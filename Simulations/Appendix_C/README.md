
# Appendix C – Extra Numerical Results

This folder contains the code, data, and results for reproducing **Appendix C** of our paper.

---

## 📄 Overview
In Appendix C, we evaluate the performance and computational cost of our proposed method under various settings. We also analyze the effect of Redistribution Matrix (RMF).
The main conclusions are that our recommended Corrected $Z$-test is orders of magnitude faster than bootstrap-based algorithms and that the Redistribution Matrix typically only have very slight influence on the final $p$-values, unless the elements in the matrix are extremely dispersed.

---

## ▶️ How to Reproduce
1. Make sure you have installed the dependencies from the main README.
   
2. Run: `Cstat_doubleB.py` (Results for makeing Figure 6), `results/Plot_time.py` (Figure 6), `RMF_vs_nonRMF_histplot.py` (Figure 7&8) and `RMFvsnoRMF.py` (Table 3).

This will:
- Load the data from `results/data_double/`
- Run the script
- Save the figures to `results/figure/` or directly print the computational time.

## 📊 Results
Example output (Figure 6 from the paper):

![Figure6](results/figure/time_boxplot.pdf)


