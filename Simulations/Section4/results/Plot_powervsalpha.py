from Plot_1valpha import *


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")


def plot_powervsalpha_single(target_params, save_prefix=None, test='one-sided'):
    full_df = load_data(target_params,test=test)
    if full_df.empty:
        print("No Matches Found!")
        return

    methods = ['Chisq', 'Plug_in', 'Cond', 'SingleB']
    method_names=[r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color

    # Type I Error ==============================================
    null_df = full_df[~full_df["is_null"]].sort_values("alpha")
    min_alpha = null_df["alpha"].min()
    max_alpha = null_df["alpha"].max()
    markers = ["s", "o", "x", "v"]

    for method, color, method_name, marker in zip(methods, colors, method_names,markers):
        # Data
        grouped = null_df.groupby("alpha", as_index=False)[method].mean()
        x = grouped["alpha"].values
        y = grouped[method].values

        # Smooth
        #if len(x) > 3:
        #    x_smooth = np.linspace(x.min(), x.max(), 300)
        #    spl = make_interp_spline(x, y, k=3)
        #    y_smooth = spl(x_smooth)
        #else:
        #    x_smooth, y_smooth = x, y

        # Main
        plt.plot(x, y,
                 linewidth=2, color=color,
                 label=method_name, marker=marker,markevery=(0,6))

        # Mark every 6
        #mask = (grouped.index % 6 == 0)  # Adjust
        #plt.scatter(x[mask], y[mask],
        #            color=color, edgecolor='white',
        #            s=60, zorder=3, linewidth=0.8,
        #            marker='o')

    # Reference line
    ref_line = np.linspace(min_alpha, max_alpha, 100)
    plt.plot(ref_line, ref_line,
             'k--', alpha=0.5, linewidth=1,
             label='y=x', zorder=1)

    #plt.xlabel("Significance level (α)")
    #plt.ylabel("Power")
    #if target_params['strue']=='brokenpowerlaw':
    #    plt.title(f"$\mu={target_params['beta'][0]}$, $k={target_params['beta'][1]}$, $k'={target_params['strength']}$, loc$=0.5$", fontsize=14, pad=20)
    #elif target_params['strue']=="spectral_line":
    #    plt.title(f"$\mu={target_params['beta'][0]}$, $k={target_params['beta'][1]}$, strength$={target_params['strength']}$, loc$=0.5$", fontsize=14, pad=20)
    #else: raise ValueError("Invalid strue!")
    plt.grid(True, alpha=0.3)
    #plt.tight_layout()



if __name__ == "__main__":
    # Fix params, brokenpowerlaw
    test='one-sided'
    beta1=[1.0,2.5,5.0,10.0]
    target_params = [{
        "n": 50,
        "beta": [i, 1.],
        "strue": "brokenpowerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 3000,
    } for i in beta1]

    params_str = format_params(target_params[0])
    plt.figure(figsize=(12, 9))
    for i in range(len(beta1)):
        plt.subplot(2, 2, i + 1)
        plt.suptitle(f"Power vs α\n({params_str})", fontsize=14)
        plot_powervsalpha_single(target_params[i],test=test)
        if i==0:
            plt.legend()
    plt.tight_layout()
    final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figurenew/{test}/{final_prefix}_powervsalpha_all.pdf", bbox_inches='tight')

    # Fix params, spec
    test='one-sided'
    beta2=[0.1, 1, 1, 5]
    strength=[1,0.1,3,1]
    target_params = [{
        "n": 50,
        "beta": [beta2[i], 1.],
        "strue": "spectral_line",
        "snull": "powerlaw",
        "strength": strength[i],
        "iters": 3000,
    } for i in range(len(beta2))]

    params_str = format_params(target_params[0])
    plt.figure(figsize=(12, 9))
    for i in range(len(beta1)):
        plt.subplot(2, 2, i + 1)
        plt.suptitle(f"Power vs α\n({params_str})", fontsize=14)
        plot_powervsalpha_single(target_params[i],test=test)
        if i==1:
            plt.legend()
    plt.tight_layout()
    final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figure/{test}/{final_prefix}_powervsalpha_all.pdf", bbox_inches='tight')
