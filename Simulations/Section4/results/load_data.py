import re
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def parse_filename(filename):

    pattern = r"results_n(\d+)_beta([\d._]+)_strue(\w+)_snull(\w+)_strength([\d.]+)_iters([\d.]+)\.xlsx"
    match = re.fullmatch(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match: {filename}")

    # beta
    beta_str = match.group(2).replace("_", " ")
    beta_values = list(map(float, beta_str.split()))

    return {
        "n": int(match.group(1)),
        "beta": beta_values,
        "strue": match.group(3),
        "snull": match.group(4),
        "strength": float(match.group(5)),
        "iters": float(match.group(6))
    }


def load_data(target_params,test="one-sided"):

    if test == "one-sided": # one-sided or two-sided
        data_dir = "data/onesided"
    elif test == "two-sided":
        data_dir = "data/twosided"
    elif test == "time_double":
        data_dir = "data_double/time"
    else:
        raise ValueError("Invalid test type! Possible types: one-sided, two-sided, time_double.")
    files = Path(data_dir).glob("results_*.xlsx")
    rows = []
    for file_path in files:
        try:
            params = parse_filename(file_path.name)
        except ValueError:
            continue

        # param match
        match = all(
            params.get(k) == v
            for k, v in target_params.items()
            if k != "strue"  and k!="strength"
        )

        if match:
            df = pd.read_excel(file_path)
            # is null?
            df["is_null"] = (params["strue"] == params["snull"])
            if params["strue"]==target_params["snull"] or (params["strue"]==target_params["strue"] and params["strength"]==target_params["strength"]):
                rows.append(df)

    return pd.concat(rows) if rows else pd.DataFrame()


def plot_separate_figures(target_params, save_prefix=None):

    def create_prefix(params):
        parts = []
        for key in params:
            value = params[key]


            if isinstance(value, list):
                val_str = "_".join(map(str, value))
            elif isinstance(value, float) and value.is_integer():
                val_str = str(int(value))
            else:
                val_str = str(value)


            clean_str = f"{key}{val_str}".replace(" ", "").lower()
            parts.append(clean_str)
        return "_".join(parts)


    final_prefix = save_prefix if save_prefix else create_prefix(target_params)


    full_df = load_data(target_params)
    if full_df.empty:
        print("No matches found.")
        return

    methods = ['Chisq', 'Plug_in', 'Cond', 'SingleB']
    colors = sns.color_palette(n_colors=len(methods))

    # Type I Error ==============================================
    plt.figure(figsize=(8, 6))
    null_df = full_df[full_df["is_null"]].sort_values("alpha")
    min_alpha = null_df["alpha"].min()
    max_alpha = null_df["alpha"].max()

    for method, color in zip(methods, colors):

        grouped = null_df.groupby("alpha", as_index=False)[method].mean()
        x = grouped["alpha"].values
        y = grouped[method].values


        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x, y


        plt.plot(x_smooth, y_smooth,
                 linewidth=2, color=color,
                 label=method, zorder=2)


        mask = (grouped.index % 6 == 0)
        plt.scatter(x[mask], y[mask],
                    color=color, edgecolor='white',
                    s=60, zorder=3, linewidth=0.8,
                    marker='o')


    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line,
             'k--', alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    plt.xlabel("Nominal significance level (α)")
    plt.ylabel("Type I Error Rate")
    plt.title("Type I Error vs α")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figure/{final_prefix}_type1_error.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Power  =====================================================
    plt.figure(figsize=(8, 6))

    merged = pd.merge(
        full_df[full_df["is_null"]].groupby("alpha")[methods].mean(),
        full_df[~full_df["is_null"]].groupby("alpha")[methods].mean(),
        left_index=True, right_index=True,
        suffixes=('_null', '_alt')
    )

    for method, color in zip(methods, colors):

        x = merged[f"{method}_null"].values
        y = merged[f"{method}_alt"].values
        alpha_values = merged.index.values


        if len(x) > 3:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]

            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            spl = make_interp_spline(x_sorted, y_sorted, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x, y


        plt.plot(x_smooth, y_smooth,
                 linewidth=2, color=color,
                 label=method, zorder=2)


        mask = (np.arange(len(x)) % 6 == 0)
        plt.scatter(x[mask], y[mask],
                    color=color, edgecolor='white',
                    s=60, zorder=3, linewidth=0.8,
                    marker='s')


    plt.plot(ref_line, ref_line, 'k--',
             alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    plt.xlabel("Type I Error Rate")
    plt.ylabel("Power")
    plt.title("Power vs Type I Error")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"test.png", dpi=300, bbox_inches='tight')
    #plt.savefig(f"figure/{final_prefix}_power_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



if __name__ == "__main__":

    target_params = {
        "n": 50,
        "beta": [5, 1.],
        "strue": "brokenpowerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 3000.0,
    }
    plot_separate_figures(target_params)