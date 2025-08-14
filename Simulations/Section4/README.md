
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
- Load the data from 'data/'
- Run the main script in 'code/'
- Save the figures to 'results/'

## 📊 Results
Example output (Figure 1 from the paper):

## Simulation Settings

In our simulation design, we set $\Gamma= 1$, and treat $n\in \{10, 25, 50, 100, 200 \}$, and $K\in \{0.1, 0.25, 0.5, 1, 1.6, 2.5, 5,10 \}$ as factors. The entire set of simulations is repeated with (a) no line, i.e., $b=0$, (b) an emission line with $\Phi = 2K$, and (c) an absorption line with $\Psi = K/10$. In both (b) and (c) the line is located with $m_1=n/2$ and $b=n/10$ so that the line extends over the energy range $(1.5, 1.6)$, but shifted one-half of a bin width to the right.

You can reproduce a specific setting by:
```bash
python code/main.py --setting S1
