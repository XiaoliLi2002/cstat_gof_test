import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
import seaborn as sns
from Simulations.Section4.results.load_data import load_data
from Simulations.Section4.results.Plot_1valpha import create_prefix, format_params
import matplotlib

# 设置全局样式
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")
matplotlib.use('TkAgg')

def plot_time_boxplot(target_params):
    full_df = load_data(target_params, test=test).drop(columns=["is_null"])
    if full_df.empty:
        print("No matches found!")
        return

    methods = ['Chisq', 'Plug_in', 'Cond', 'SingleB', 'DoubleB']
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test (Alg.2c)', r'Corrected Z-test (Alg.3b)', r'Parametric Bootstrap', r'Double Bootstrap']
    full_df.columns = method_names
    df_long = full_df.melt(var_name='Method', value_name='Time')
    df_long["Time"] = df_long["Time"] + 1e-3
    print(df_long)

    plt.figure(figsize=(16, 9))
    sns.boxplot(x="Method", y="Time", data=df_long, palette="Set2")
    plt.yscale("log")
    plt.title(f"$n=100$, $K=1$, $\Gamma=1$", fontsize=14)
    plt.ylabel("Time in log-scale (s)", fontsize=14)
    plt.xlabel("Method", fontsize=14)
    plt.savefig(Path("figure/time_boxplot.pdf"), bbox_inches='tight')


if __name__ == "__main__":

    test = 'time_double'  # one-sided or two-sided
    beta1 = [1.0]
    target_params = [{
        "n": 100,
        "beta": [i, 1.],
        "strue": "powerlaw",
        "snull": "powerlaw",
        "strength": 0.1,
        "iters": 100,
    } for i in beta1]

    plot_time_boxplot(target_params[0])