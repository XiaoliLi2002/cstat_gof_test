import matplotlib.pyplot as plt

from Simulations.Section4.results.load_data import *


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def create_prefix(params):
    parts = []
    for key in params:
        if key=="beta":
            continue
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

def format_params(params):

    parts = []
    for k, v in params.items():
        if k == "strue_snull_relation" or k=="beta" or k=="strength":
            continue

        if isinstance(v, float):
            if v.is_integer():
                val_str = str(int(v))
            else:
                val_str = f"{v:.2f}".rstrip('0').rstrip('.')
        elif isinstance(v, list):
            val_str = ", ".join([f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else str(x) for x in v])
        else:
            val_str = str(v)
        parts.append(f"{k}={val_str}")
    return ", ".join(parts)

def plot_1vsalpha_single(target_params, save_prefix=None, test='one-sided'):
    full_df = load_data(target_params,test=test)
    if full_df.empty:
        print("No matches found!")
        return

    methods = ['Chisq', 'Plug_in', 'Cond', 'SingleB']
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color

    # Type I Error ==============================================
    null_df = full_df[full_df["is_null"]].sort_values("alpha")
    min_alpha = null_df["alpha"].min()
    max_alpha = null_df["alpha"].max()
    markers = ["s", "o", "x", "v"]

    for method, color, method_name, marker in zip(methods, colors, method_names,markers):

        grouped = null_df.groupby("alpha", as_index=False)[method].mean()
        x = grouped["alpha"].values
        y = grouped[method].values


        #if len(x) > 3:
        #    x_smooth = np.linspace(x.min(), x.max(), 300)
        #    spl = make_interp_spline(x, y, k=3)
        #    y_smooth = spl(x_smooth)
        #else:
        #    x_smooth, y_smooth = x, y


        plt.plot(x, y,
                 linewidth=2, color=color,
                 label=method_name, marker=marker,markevery=(0,6))


        #mask = (grouped.index % 6 == 0)
        #plt.scatter(x[mask], y[mask],
        #            color=color, edgecolor='white',
        #            s=60, zorder=3, linewidth=0.8,
        #            marker=marker)


    #plt.xlabel("Significance level (α)")
    plt.title(f"$K={target_params['beta'][0]}$",fontsize=16)
    #plt.title(f"$\mu={target_params['beta'][0]}$, $k={target_params['beta'][1]}$", fontsize=14, pad=20)
    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line,
             'k--', alpha=0.5, linewidth=1,
             zorder=1)

    plt.grid(True, alpha=0.3)



if __name__ == "__main__":

    test='one-sided' # one-sided or two-sided
    beta1=[0.1, 1.0,2.5,5.0]
    target_params = [{
        "n": 50,
        "beta": [i, 1.],
        "strue": "powerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 3000,
    } for i in beta1]

    params_str = format_params(target_params[0])
    plt.figure(figsize=(12, 9))
    for i in range(len(beta1)):
        plt.subplot(2,2,i+1)
        plt.suptitle(f"Type I Error vs α\n({params_str})", fontsize=14)
        plot_1vsalpha_single(target_params[i], test=test)
        if i==0:
            plt.legend()
    
    plt.tight_layout()
    final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figure/{test}/{final_prefix}_type1vsalpha_all.pdf", bbox_inches='tight')

