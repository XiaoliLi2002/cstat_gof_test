import matplotlib.pyplot as plt
import numpy as np

from Simulations.Section4.results.Plot_1valpha import plot_1vsalpha_single
from Simulations.Section4.results.Plot_powervsalpha import *
from Simulations.utilities.utilities import generate_s_true


def plot_spec_counts(n,beta,loc,strength1,strength2,width,alpha=0.8):
    plt.ylim([-0.1,10.2])
    energy = np.linspace(1 + 1 / n, 2, n)
    s1=generate_s_true(n,beta,'powerlaw','powerlaw')
    s2=generate_s_true(n,beta,'spectral_line','powerlaw',loc=loc,strength=strength1,width=width)
    s3=generate_s_true(n,beta,'spectral_line','powerlaw',loc=loc,strength=strength2,width=width)

    plt.plot(energy, s1, color="black", label=r'No Line', alpha=alpha)
    #plt.scatter(energy, x1, color=colors[0], label=r'Count, powerlaw', alpha=alpha,marker=markers[0],s=size)

    plt.plot(energy, s2, color="#D55E00", label=r'Absorption Line', alpha=alpha)
    #plt.scatter(energy, x2, color=colors[1], label=r'Count, emission', alpha=alpha, marker=markers[1],s=size)

    plt.plot(energy, s3, color="#56B4E9", label=r'Emission Line', alpha=alpha)
    #plt.scatter(energy, x3, color=colors[2], label=r'Count, absorption', alpha=alpha, marker=markers[2],s=size)

    plt.yticks(fontsize=18)
    plt.xlabel(r'$E_i$', fontsize=20)
    #plt.title(f'Powerlaw, $n={n},K={beta[0]},\Gamma={beta[1]},s_+={round(np.sum(s1))},N_+={round(np.sum(x1))}$', fontsize=16)
    plt.grid(linestyle='--', alpha=0.5)


if __name__ == "__main__":

    test = 'one-sided'  # one-sided or two-sided
    n = 50
    beta1 = np.array([0.25, 1.0, 5.0])
    strength_absorb = beta1/10
    strength_emission = beta1*2
    loc=0.5
    width = int(n / 10)

    target_params_type1 = [{
        "n": 50,
        "beta": [i, 1.],
        "strue": "powerlaw",
        "snull": "powerlaw",
        "strength": 3,
        "iters": 3000,
    } for i in beta1]

    target_params_power_absorb = [{
        "n": 50,
        "beta": [beta1[i], 1.],
        "strue": "spectral_line",
        "snull": "powerlaw",
        "strength": strength_absorb[i],
        "iters": 3000,
    } for i in range(len(beta1))]

    target_params_power_emission = [{
        "n": 50,
        "beta": [beta1[i], 1.],
        "strue": "spectral_line",
        "snull": "powerlaw",
        "strength": strength_emission[i],
        "iters": 3000,
    } for i in range(len(beta1))]

    #params_str = format_params(target_params[0])
    plt.figure(figsize=(15, 17))
    for i in range(len(beta1)):
        plt.subplot(4, 3, i + 1)
        #plt.suptitle(f"Type I Error vs α\n({params_str})", fontsize=14)
        plot_1vsalpha_single(target_params_type1[i], test=test)
        if i == 0:
            plt.legend(prop={'size': 15})
            plt.ylabel("Type I Error",fontsize=20)
        #if i == 2:
        #    xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        #    xlabels= ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        #    plt.xticks(xticks,xlabels)
        #else:
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels = ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        plt.xticks(xticks, xlabels, fontsize=18)
        plt.xlabel(r"Significance level ($\alpha$)", fontsize=20)


    for i in range(len(beta1)):
        plt.subplot(4, 3, i + 4)
        plot_powervsalpha_single(target_params_power_absorb[i], test=test)
        if i == 0:
            #plt.legend()
            plt.ylabel(f"Power, $\Psi/K=0.1$",fontsize=20)
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels = ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        plt.xticks(xticks, xlabels, fontsize=18)
        plt.xlabel(r"Significance level ($\alpha$)", fontsize=20)

    for i in range(len(beta1)):
        plt.subplot(4, 3, i + 7)
        plot_powervsalpha_single(target_params_power_emission[i], test=test)
        if i == 0:
            #plt.legend()
            plt.ylabel(f"Power, $\Psi/K=2$",fontsize=20)
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        xlabels = ['0', '0.05', '0.10', '0.15', '0.20', '0.25']
        plt.xticks(xticks, xlabels, fontsize=18)
        plt.xlabel(r"Significance level ($\alpha$)", fontsize=20)

    for i in range(len(beta1)):
        plt.subplot(4, 3, i + 10)
        plot_spec_counts(n=n,beta=target_params_power_emission[i]["beta"],loc=loc,width=width,strength1=strength_absorb[i],strength2=strength_emission[i])
        plt.xticks(fontsize=18)
        if i == 0:
            plt.legend(loc="upper left",prop={'size': 15})
            plt.ylabel(r"Rate",fontsize=20)



    plt.tight_layout()
    #final_prefix = create_prefix(target_params[0])
    plt.savefig(f"figure/Figure12_Gamma=1.pdf", bbox_inches='tight')
