# High-Energy Astronomy Goodness-of-Fit (HEAGOF)


We developed **HEAGOF** for the broader astronomical community and other researchers that need goodness-of-fit assessments for heterogeneous low-count Poisson data.

**HEAGOF** enables researchers to directly obtain well-calibrated principled goodness-of-fit assessment in a straightforward manner without needing a full understanding of the technical details required either to determine an appropriate approximation to the null distribution or to define and fit a fully Bayesian model.

This repository contains the package **HEAGOF** and the code, data, and results for reproducing the simulations in our paper:

> **Title:** Making high-order asymptotics practical: correcting goodness-of-fit test for astronomical count data
> 
> **Authors:** Xiaoli Li, Yang Chen, Xiao-Li Meng, David A. van Dyk, Massimiliano Bonamente, Vinay L. Kashyap
> 
> **Link:** [arXiv / Journal link] 

---

## 📂 Structure
- `HEAGOF` – the package **HEAGOF**
- `simulations` – the code, data, and results for reproducing the simulations in our paper

---

## ⚙️ Installation

git clone https://github.com/yourusername/yourrepo.git

cd yourrepo

pip install -r requirements.txt

---

## 🚀 Quick Start
Example for using our package to test a Powerlaw model:

`cd HEAGOF`

run `main.py`

This will run our package to test a synthetic powerlaw spectrum with a powerlaw model.

For more details, see `HEAGOF/README.md`.

---


## 📜 Citation

If you use this code, please cite:

@article{your_paper_citation,

  title={Your Paper Title},
  
  author={Your Name and Coauthor Name},
  
  journal={Journal Name},
  
  year={2025}
  
}

