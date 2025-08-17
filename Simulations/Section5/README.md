
# Section 5 – Applications to Astrophysical Data

This folder contains the code, data, and results for reproducing **Section 5** of our paper.

---

## 📄 Overview
In Section 5, we use our methods to test whether the constant model is sufficient for each of the four spectra.
The main conclusion is that commonly used default $\chi^2$ approximation gives severe false acceptances and rejections.

---

## ▶️ How to Reproduce
1. Make sure you have installed the dependencies from the main README.
   
2. Run: `pg1116Spectra.py` (Figure 5, left panel), `Constant_fit_test.py` (Table 2) and `pg1116-3145-20/Constant_fitted.py` (Results for one single segment of 20 segments).

This will:
- Load the data from `countFormatxxx.dat` or `pg1116-3145-20/countFormatxxx-segment-xxx-20.dat`, where "xxx" represents digits.
- Run the script
- Then the script will directly print the figure or results.


