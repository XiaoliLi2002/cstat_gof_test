import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from scipy.stats import chi2
from Plot_powervsalpha import *


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
sns.set_palette("husl")

def load_data_new(target_params):
    """Load data"""

    # Load CriticalValue
    #print(target_params)
    data_dir = "data/CriticalValue"
    files = Path(data_dir).glob("results_*.xlsx")
    criticalvalue_loaded_list=[]
    for file_path in files:
        try:
            params = parse_filename(file_path.name)
        except ValueError:
            continue

        #
        match = all(
            params.get(k) == v
            for k, v in target_params.items()
            if k == "beta" or k=='n' or k=='strue'
        )

        if match:
            df = pd.read_excel(file_path).to_numpy()
            criticalvalue_loaded_list.append(df)

    # Load Width
    data_dir = "data/Width"
    files = Path(data_dir).glob("results_*.xlsx")
    width_loaded_list=[]
    for file_path in files:
        try:
            params = parse_filename(file_path.name)
        except ValueError:
            continue

        # 检查参数匹配
        match = all(
            params.get(k) == v
            for k, v in target_params.items()
            if k == "beta" or k=='n' or k=='strue'
        )

        if match:
            df = pd.read_excel(file_path).to_numpy()
            width_loaded_list.append(df)

    return np.stack(criticalvalue_loaded_list).squeeze(), np.stack(width_loaded_list).squeeze()

def plot_covvsn_single(df,mu,alpha=0.10):
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers=["s", "o", "x", "v"]

    for i in range(len(method_names)):
        plt.plot(n_params, df[:,i], label=method_names[i], color=colors[i], marker=markers[i], alpha=0.8)


    plt.ylim((-0.02,max(0.37,df.max()+0.05)))
    #plt.xlim((7,500))
    plt.xlim((7,500))
    plt.xscale("log")
    xtick_labels=["10","25","50","100","200","","400"]
    #xtick_labels = ["10", "25", "50", "100", "200"]
    if mu==0.25:
        xtick_labels=["0.9375","2.34375","4.6875","9.375","18.75","","37.5"]
    elif mu==2.5:
        xtick_labels = ["9.375", "23.4375", "46.875", "93.75", "187.5", "", "375"]
    plt.xticks(n_params, xtick_labels)
    ref_line = np.linspace(7, 500, 100)
    plt.plot(ref_line, (alpha) * np.ones(len(ref_line)),
             'k--', alpha=0.5, linewidth=1,
             zorder=1)
    plt.title(f'Powerlaw, $K={mu},\Gamma=3, n\in[10,400]$')

    plt.grid(True, alpha=0.3)

def plot_cricvsn_single(df,mu,alpha=0.10):
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers = ["-s", "-o", "-x", "-v"]

    for i in range(len(method_names)):
        if i==0:
            plt.errorbar(n_params,chi2.isf(alpha,n_params-2)/n_params,yerr=np.zeros(len(n_params)),fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])
            #plt.errorbar(n_params, chi2.isf(alpha, n_params - 2), yerr=np.zeros(len(n_params)), fmt=markers[i], alpha=0.8, capsize=5)
        else:
            plotdata=df[:,:,i]
            for j in range(len(n_params)):
                plotdata[j,:]=plotdata[j,:]/n_params[j]
            average=plotdata.mean(axis=1)
            errorlimit=[average-np.quantile(plotdata,q=0.05,axis=1),np.quantile(plotdata,q=0.95,axis=1)-average]
            plt.errorbar(n_params,average,yerr=errorlimit,fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])

    # plt.xlim((7,500))
    plt.xlim((7, 500))
    plt.xscale("log")
    xtick_labels = ["10", "25", "50", "100", "200", "", "400"]
    # xtick_labels = ["10", "25", "50", "100", "200"]
    if mu == 0.25:
        xtick_labels = ["0.9375", "2.34375", "4.6875", "9.375", "18.75", "", "37.5"]
    elif mu == 2.5:
        xtick_labels = ["9.375", "23.4375", "46.875", "93.75", "187.5", "", "375"]
    plt.xticks(n_params, xtick_labels)
    #plt.yticks(n_params, xtick_labels)

    plt.grid(True, alpha=0.3)

def plot_widthvsn_single(df,mu,alpha=0.10):
    method_names = [r'Alg.1: LR-$\chi^2$ test', r'Alg.2c: Naive Z-test', r'Alg.3b: Corrected Z-test', r'Alg.4: Param. Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers = ["-s", "-o", "-x", "-v"]

    for i in range(len(method_names)):
        if i==0:
            plt.errorbar(n_params,chi2.isf(alpha/2,n_params-2)-chi2.isf(1-alpha/2,n_params-2),yerr=np.zeros(len(n_params)),fmt=markers[i], alpha=0.8,capsize=5)
        else:
            plotdata=df[:,:,i]
            average=plotdata.mean(axis=1)
            errorlimit=[average-np.quantile(plotdata,q=0.05,axis=1),np.quantile(plotdata,q=0.95,axis=1)-average]
            plt.errorbar(n_params,average,yerr=errorlimit,fmt=markers[i], alpha=0.8,capsize=5,color=colors[i])

    plt.xlim((7, 500))
    #plt.ylim((2,500))
    plt.xscale("log")
    #plt.yscale("log")
    xtick_labels = ["10", "25", "50", "100", "200", "", "400"]
    # xtick_labels = ["10", "25", "50", "100", "200"]
    if mu == 0.25:
        xtick_labels = ["0.9375", "2.34375", "4.6875", "9.375", "18.75", "", "37.5"]
    elif mu == 2.5:
        xtick_labels = ["9.375", "23.4375", "46.875", "93.75", "187.5", "", "375"]
    plt.xticks(n_params, xtick_labels)
    #plt.yticks(n_params, xtick_labels)

    plt.grid(True, alpha=0.3)


if __name__=='__main__':
    # Output figure: 2x3: Coverage, CriticalValue, Width; mu=0.25, 2.5; k=1.
    # Coverage no error bar (Binomial dist. <+-0.02); CriticalValue&Width Error bar

    # data
    type1error1=np.array(  # type I error at beta1=0.1 mu=0.1
        [
            [0,	0.00067,	0.01233,	0.00967],
            [0,	0.001,	0.04133,	0.00267],
            [0,	0,	0.06667,	0.00033],
            [0,	0,	0.09833,	0],
            [0,	0,	0.117,	0],
            #[0,	0,	0.11167,	0],
            #[0,	0,	0.115,	0]
        ]
    )

    type1error3=np.array(  # type I error at mu=0.25
        [
            [0,	0,	0.01667,	0.01233],
            [0,	0.00133,	0.05567,	0.006],
            [0,	0.00033,	0.084,	0.002],
            [0,	0,	0.10667,	0],
            [0,	0,	0.098,	0],
            [0,	0,	0.10367,	0],
            [0,	0,	0.107,	0]
        ]
    )

    type1error2=np.array(  # type I error at mu=2.5
        [
            [0.05867,	0.028,	0.07367,	0.1],
            [0.085,	0.051,	0.09867,	0.10567],
            [0.10767,	0.07,	0.108,	0.108],
            [0.138,	0.07233,	0.105,	0.10233],
            [0.20667,	0.08533,	0.108,	0.102],
            [0.244,	0.082,	0.10367,	0.095],
            [0.30533,	0.085,	0.10233,	0.099]
        ]
    )

    # Params, alpha=0.10
    n_params = np.array([10, 25, 50, 100, 200, 300, 400])
    #n_params = np.array([10, 25, 50, 100, 200])
    beta1_params = np.array([0.25, 2.5])
    alpha=0.1

    target_params = [[{
        "n": j,
        "beta": [i, 3.],  # 自动匹配beta参数
        "strue": "powerlaw",
        "snull": "powerlaw",
        "iters": 3000,
    } for i in beta1_params] for j in n_params]

    criticalvalue1 = []  # 0.25
    width1 = []
    for i in range(len(n_params)):
        a, b=load_data_new(target_params[i][0])
        criticalvalue1.append(a)
        width1.append(b)
    criticalvalue1=np.stack(criticalvalue1)
    width1=np.stack(width1)

    criticalvalue2 = []  # 2.5
    width2 = []
    for i in range(len(n_params)):
        a, b=load_data_new(target_params[i][1])
        criticalvalue2.append(a)
        width2.append(b)
    criticalvalue2=np.stack(criticalvalue2)
    width2=np.stack(width2)

    coverage = [1-type1error3,1-type1error2]
    type1=[type1error3,type1error2]
    criticalvalues= [criticalvalue1,criticalvalue2]
    widths = [width1,width2]

    fig = plt.figure(figsize=(12, 9))
    for i in range(2):

        # coverage vs n
        plt.subplot(2, 2, i + 1)
        plot_covvsn_single(type1[i],mu=beta1_params[i],alpha=0.1)
        if i == 0:
            plt.ylabel("Type I Error", fontsize=16)
            plt.legend(loc='best')

        # criticalvalue vs n
        plt.subplot(2, 2, i + 3)
        if i == 0:
            plt.ylabel(r"Critical Value / $n$", fontsize=16)
            #plt.title(r"Critical Value")
        #plt.xlabel(r"Number of Bins $(n)$", fontsize=16)
        plt.xlabel(f"Total Expected Counts $s_+$", fontsize=16)
        plot_cricvsn_single(criticalvalues[i],mu=beta1_params[i],alpha=0.1)

        # width vs n
        #plt.subplot(2, 3, i * 3 + 3)
        #if i == 0:
        #    plt.title("Width")
        #plot_widthvsn_single(widths[i],alpha)


    plt.tight_layout()
    plt.savefig(f"figure/Figure3_Gamma=3.pdf", bbox_inches='tight')