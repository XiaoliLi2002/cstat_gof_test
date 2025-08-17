import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

from scipy.stats import chi2
from Simulations.Section4.results.Plot_powervsalpha import *
from Plot_cov_width_n import load_data_new


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")

def plot_covvsmu_single(df,n,alpha=0.10):
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers=["s", "o", "x", "v"]

    for i in range(len(method_names)):
        plt.plot(mu_params, df[:,i], label=method_names[i], color=colors[i], marker=markers[i], alpha=0.8)


    plt.ylim((-0.02,max(0.22,df.max()+0.05)))
    plt.xlim((0.07,7))
    plt.xscale("log")
    xtick_labels=["0.1","0.25","0.5","1","1.6","2.5","5"]
    plt.xticks(mu_params, xtick_labels)
    ref_line = np.linspace(0.07, 7, 100)
    plt.plot(ref_line, (alpha) * np.ones(len(ref_line)),
             'k--', alpha=0.5, linewidth=1,
             zorder=1)
    plt.title(f'$n={n}$')

    plt.grid(True, alpha=0.3)

def plot_cricvsmu_single(df,n,alpha=0.10):
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers = ["-s", "-o", "-x", "-v"]
    n_params=np.array([n for i in range(len(mu_params))])

    for i in range(len(method_names)):
        if i==0:
            plt.errorbar(mu_params,chi2.isf(alpha,n_params-2)/n_params,yerr=np.zeros(len(n_params)),fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])
            #print(chi2.isf(alpha,n_params-2)/n_params)
            #plt.errorbar(mu_params, chi2.isf(alpha, n_params - 2), yerr=np.zeros(len(n_params)),fmt=markers[i], alpha=0.8, capsize=5)
        else:
            plotdata=df[:,:,i]
            for j in range(len(n_params)):
                plotdata[j,:]=plotdata[j,:]/n_params[j]
            average=plotdata.mean(axis=1)
            errorlimit=[average-np.quantile(plotdata,q=0.05,axis=1),np.quantile(plotdata,q=0.95,axis=1)-average]
            plt.errorbar(mu_params,average,yerr=errorlimit,fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])

    #plt.ylim((0.18,2.3))
    #yticks=[0.2, 0.3, 0.4, 0.6, 1, 1.5, 2]
    #ylabels=['0.2', '0.3', '0.4', '0.6','1','1.5','2']
    #plt.ylim((n/5.5,1.5*n))
    #if n==10:
    #    yticks=[2,2.5, 5, 7.5, 10, 15, 25]
    #    ylabels=['2','2.5','5','7.5','10','15','25']
    #elif n==50:
    #    yticks=[10, 20, 30, 40, 50, 60, 75]
    #    ylabels=['10', '20', '30', '40' ,'50','60', '75']
    #else:
    #    yticks=np.logspace(n/5,1.5*n,5)
    #    ylabels=[]
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlim((0.07, 7))
    plt.xscale("log")
    xtick_labels = ["0.1", "0.25", "0.5", "1", "1.6", "2.5", "5"]
    plt.xticks(mu_params, xtick_labels)
    #plt.yticks(yticks,ylabels)

    plt.grid(True, alpha=0.3)

def plot_widthvsmu_single(df,n,alpha=0.10):
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers = ["-s", "-o", "-x", "-v"]
    n_params=np.array([n for i in range(len(mu_params))])


    for i in range(len(method_names)):
        if i==0:
            plt.errorbar(mu_params,chi2.isf(alpha/2,n_params-2)-chi2.isf(1-alpha/2,n_params-2),yerr=np.zeros(len(n_params)),fmt=markers[i], alpha=0.8,capsize=5)
        else:
            plotdata=df[:,:,i]
            average=plotdata.mean(axis=1)
            errorlimit=[average-np.quantile(plotdata,q=0.05,axis=1),np.quantile(plotdata,q=0.95,axis=1)-average]
            plt.errorbar(mu_params,average,yerr=errorlimit,fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])

    #plt.ylim((2,500))
    plt.xscale("log")
    plt.xlim((0.07, 7))
    plt.xscale("log")
    xtick_labels = ["0.1", "0.25", "0.5", "1", "1.6", "2.5", "5"]
    plt.xticks(mu_params, xtick_labels)

    plt.grid(True, alpha=0.3)


if __name__=='__main__':
    # Output figure: 2x3: Coverage, CriticalValue, Width; mu=0.1, 0.25, 0.5, 1, 1.6, 2.5, 5; k=1. n=10, 50
    # Coverage no error bar (Binomial dist. <+-0.02); CriticalValue&Width Error bar

    # data
    type1error1=np.array(  # type I error at n=10
        [
            [0,	0.00067,	0.01233,	0.00967],
            [0.00133,	0.00133,	0.03833,	0.03],
            [0.01533,	0.01467,	0.072,	0.075],
            [0.064,	0.02933,	0.091,	0.11233],
            [0.12767,	0.04267,	0.09367,	0.11867],
            [0.15867,	0.04267,	0.09033,	0.105],
            [0.14867,	0.047,	0.09033,	0.09667]
        ]
    )

    type1error2=np.array(  # type I error at n=50
        [
            [0,	0,	0.06667,	0.00033],
            [0,	0.00367,	0.09667,	0.01333],
            [0.002,	0.021,	0.088,	0.05067],
            [0.10833,	0.066,	0.109,	0.10833],
            [0.24133,	0.07033,	0.09867,	0.10133],
            [0.26567,	0.06333,	0.09167,	0.09367],
            [0.20033,	0.071,	0.10333,	0.09967]
        ]
    )

    # Params, alpha=0.10
    #n_params = np.array([10, 25, 50, 100, 200, 300, 400])
    #beta1_params = np.array([0.1, 2.5])
    mu_params=np.array([0.1, 0.25, 0.5, 1, 1.6, 2.5, 5])
    n_param=np.array([10,50])
    alpha=0.1

    target_params = [[{
        "n": i,
        "beta": [j, 1.],
        "strue": "powerlaw",
        "snull": "powerlaw",
        "iters": 3000,
    } for i in n_param] for j in mu_params]

    criticalvalue1 = []  # 10
    width1 = []
    for i in range(len(mu_params)):
        a, b=load_data_new(target_params[i][0])
        criticalvalue1.append(a)
        width1.append(b)
    criticalvalue1=np.stack(criticalvalue1)
    width1=np.stack(width1)

    criticalvalue2 = []  # 50
    width2 = []
    for i in range(len(mu_params)):
        a, b=load_data_new(target_params[i][1])
        criticalvalue2.append(a)
        width2.append(b)
    criticalvalue2=np.stack(criticalvalue2)
    width2=np.stack(width2)

    coverage = [1-type1error1,1-type1error2]
    type1=[type1error1,type1error2]
    criticalvalues= [criticalvalue1,criticalvalue2]
    widths = [width1,width2]

    fig = plt.figure(figsize=(12, 9))
    for i in range(2):

        # coverage vs n
        plt.subplot(2, 2, i + 1)
        plot_covvsmu_single(type1[i],n=n_param[i],alpha=0.1)
        #plt.ylim((0.72,1.02))
        plt.ylim((-0.02,0.28))
        if i == 0:
            plt.ylabel("Type I Error", fontsize=16)
            plt.legend(loc='best')

        # criticalvalue vs n
        plt.subplot(2, 2, i + 3)
        if i == 0:
            plt.ylabel(r"Critical Value / $n$", fontsize=16)
            #plt.title(r"Critical Value")
        plt.xlabel(f"$K$", fontsize=16)
        plot_cricvsmu_single(criticalvalues[i],n=n_param[i],alpha=alpha)

        ## width vs n
        #plt.subplot(2, 3, i * 3 + 3)
        #if i == 0:
        #    plt.title("Width")
        #plot_widthvsmu_single(widths[i],n=n_param[i],alpha=alpha)


    plt.tight_layout()
    plt.savefig(f"figure/2x3vsmu.pdf", bbox_inches='tight')