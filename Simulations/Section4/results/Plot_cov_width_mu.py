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
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers=["s", "o", "x", "v"]

    for i in range(len(method_names)):
        plt.plot(mu_params, df[:,i], label=method_names[i], color=colors[i], marker=markers[i], alpha=0.8)


    plt.ylim((-0.02,max(0.32,df.max()+0.02)))
    plt.xlim((0.07,12))
    plt.xscale("log")
    xtick_labels=["0.1","0.25","0.5","1","1.6","2.5","5","10"]
    if n==10:
        xtick_labels=["0.375","0.9375","1.875","3.75","6","9.375","18.75","37.5"]
    elif n==100:
        xtick_labels = ["3.75", "9.375", "18.75", "37.5", "60", "93.75", "187.5", "375"]
    plt.xticks(mu_params, xtick_labels)
    ref_line = np.linspace(0.07, 12, 100)
    plt.plot(ref_line, (alpha) * np.ones(len(ref_line)),
             'k--', alpha=0.5, linewidth=1,
             zorder=1)
    plt.title(f'Powerlaw, $n={n},\Gamma=3,K\in[0.1,10]$')

    plt.grid(True, alpha=0.3)

def plot_cricvsmu_single(df,n,alpha=0.10):
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
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
    plt.xlim((0.07, 12))
    plt.xscale("log")
    xtick_labels = ["0.1", "0.25", "0.5", "1", "1.6", "2.5", "5", "10"]
    if n == 10:
        xtick_labels = ["0.375", "0.9375", "1.875", "3.75", "6", "9.375", "18.75", "37.5"]
    elif n == 100:
        xtick_labels = ["3.75", "9.375", "18.75", "37.5", "60", "93.75", "187.5", "375"]
    plt.xticks(mu_params, xtick_labels)
    #plt.yticks(yticks,ylabels)

    plt.grid(True, alpha=0.3)

def plot_widthvsmu_single(df,n,alpha=0.10):
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
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
    plt.xlim((0.07, 12))
    plt.xscale("log")
    xtick_labels = ["0.1", "0.25", "0.5", "1", "1.6", "2.5", "5", "10"]
    if n == 10:
        xtick_labels = ["0.375", "0.9375", "1.875", "3.75", "6", "9.375", "18.75", "37.5"]
    elif n == 100:
        xtick_labels = ["3.75", "9.375", "18.75", "37.5", "60", "93.75", "187.5", "375"]
    plt.xticks(mu_params, xtick_labels)

    plt.grid(True, alpha=0.3)


if __name__=='__main__':
    # Output figure: 2x3: Coverage, CriticalValue, Width; mu=0.1, 0.25, 0.5, 1, 1.6, 2.5, 5; k=1. n=10, 50
    # Coverage no error bar (Binomial dist. <+-0.02); CriticalValue&Width Error bar

    # data
    type1error1=np.array(  # type I error at n=10
        [
            [0	,0,	0.003,	0.003],
            [0,	0,	0.01667,	0.01233],
            [0.00133,	0.005,	0.04,	0.02967],
            [0.00833,	0.00867,	0.057,	0.05833],
            [0.025,	0.02,	0.08333,	0.09667],
            [0.05867,	0.028,	0.07367,	0.1],
            [0.14167,	0.04267,	0.09267,	0.11067],
            [0.16067,	0.04833,	0.095,	0.109]
        ]
    )

    type1error2=np.array(  # type I error at n=100
        [
            [0,	0,	0.06233,	0],
            [0,	0,	0.10667,	0],
            [0,	0.00267,	0.09,	0.00933],
            [0,	0.02467,	0.101,	0.05133],
            [0.01733,	0.05467,	0.10233,	0.08467],
            [0.138,	0.07233,	0.105,	0.10233],
            [0.30767,	0.08167	,0.10167,	0.10067],
            [0.27533,	0.08267,	0.10567,	0.10733]
        ]
    )

    # Params, alpha=0.10
    #n_params = np.array([10, 25, 50, 100, 200, 300, 400])
    #beta1_params = np.array([0.1, 2.5])
    mu_params=np.array([0.1, 0.25, 0.5, 1, 1.6, 2.5, 5,10])
    n_param=np.array([10,100])
    alpha=0.1

    target_params = [[{
        "n": i,
        "beta": [j, 3.],  # 自动匹配beta参数
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
        #plt.ylim((-0.02,0.32))
        if i == 0:
            plt.ylabel("Type I Error", fontsize=16)
            plt.legend(loc='best')

        # criticalvalue vs n
        plt.subplot(2, 2, i + 3)
        if i == 0:
            plt.ylabel(r"Critical Value / $n$", fontsize=16)
            #plt.title(r"Critical Value")
        plt.xlabel(f"Total Expected Counts $s_+$", fontsize=16)
        plot_cricvsmu_single(criticalvalues[i],n=n_param[i],alpha=alpha)

        ## width vs n
        #plt.subplot(2, 3, i * 3 + 3)
        #if i == 0:
        #    plt.title("Width")
        #plot_widthvsmu_single(widths[i],n=n_param[i],alpha=alpha)


    plt.tight_layout()
    plt.savefig(f"figure/Figure2_Gamma=3.pdf", bbox_inches='tight')