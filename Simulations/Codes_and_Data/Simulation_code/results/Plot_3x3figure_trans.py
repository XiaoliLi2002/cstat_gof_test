import matplotlib.pyplot as plt
import numpy as np

from Plot_1valpha import plot_1vsalpha_single
from Plot_powervsalpha import *

# 使用示例
if __name__ == "__main__":
    # 指定需要固定的参数
    test = 'one-sided'  # one-sided or two-sided
    beta1 = np.array([0.25, 1.0, 5.0])
    strength_absorb = beta1/10
    strength_emission = beta1*2

    target_params_type1 = [{
        "n": 50,
        "beta": [i, 1.],  # 自动匹配beta参数
        "strue": "powerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 3000,
    } for i in beta1]

    target_params_power_absorb = [{
        "n": 50,
        "beta": [beta1[i], 1.],  # 自动匹配beta参数
        "strue": "spectral_line",
        "snull": "powerlaw",
        "strength": strength_absorb[i],
        "iters": 3000,
    } for i in range(len(beta1))]

    target_params_power_emission = [{
        "n": 50,
        "beta": [beta1[i], 1.],  # 自动匹配beta参数
        "strue": "spectral_line",
        "snull": "powerlaw",
        "strength": strength_emission[i],
        "iters": 3000,
    } for i in range(len(beta1))]

    #params_str = format_params(target_params[0])
    plt.figure(figsize=(15, 12))
    for i in range(len(beta1)):
        plt.subplot(3, 3, i + 1)
        #plt.suptitle(f"Type I Error vs α\n({params_str})", fontsize=14)
        plot_1vsalpha_single(target_params_type1[i], test=test)
        if i == 0:
            plt.legend()
            plt.ylabel("Type I Error",fontsize=16)
        #if i == 2:
        #    xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        #    xlabels= ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        #    plt.xticks(xticks,xlabels)
        #else:
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels = ['', '', '', '', '', '']
        plt.xticks(xticks, xlabels)


    for i in range(len(beta1)):
        plt.subplot(3, 3, i + 4)
        plot_powervsalpha_single(target_params_power_absorb[i], test=test)
        if i == 0:
            #plt.legend()
            plt.ylabel(f"Power, $\Psi/K=0.1$",fontsize=16)
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels = ['', '', '', '', '', '']
        plt.xticks(xticks, xlabels)

    for i in range(len(beta1)):
        plt.subplot(3, 3, i + 7)
        plot_powervsalpha_single(target_params_power_emission[i], test=test)
        if i == 0:
            #plt.legend()
            plt.ylabel(f"Power, $\Psi/K=2$",fontsize=16)
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels= ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        plt.xticks(xticks,xlabels)
        plt.xlabel(r"Significance level ($\alpha$)",fontsize=16)



    plt.tight_layout()
    #final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figurenew/3x3figure_trans_blackcolor.pdf", bbox_inches='tight')
