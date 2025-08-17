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
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
    colors = ["#F0E442", "#D55E00", "black", "#56B4E9"]  # color
    markers=["s", "o", "x", "v"]

    for i in range(len(method_names)):
        plt.plot(n_params, df[:,i], label=method_names[i], color=colors[i], marker=markers[i], alpha=0.8)


    plt.ylim((-0.02,max(0.22,df.max()+0.05)))
    #plt.xlim((7,500))
    plt.xlim((7,250))
    plt.xscale("log")
    #xtick_labels=["10","25","50","100","200","","400"]
    xtick_labels = ["10", "25", "50", "100", "200"]
    plt.xticks(n_params, xtick_labels)
    ref_line = np.linspace(7, 500, 100)
    plt.plot(ref_line, (alpha) * np.ones(len(ref_line)),
             'k--', alpha=0.5, linewidth=1,
             zorder=1)
    plt.title(f'$K={mu}$')

    plt.grid(True, alpha=0.3)

def plot_cricvsn_single(df,alpha=0.10):
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
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
    plt.xlim((7, 250))
    plt.xscale("log")
        # xtick_labels=["10","25","50","100","200","","400"]
    xtick_labels = ["10", "25", "50", "100", "200"]
    plt.xticks(n_params, xtick_labels)
    #plt.yticks(n_params, xtick_labels)

    plt.grid(True, alpha=0.3)

def plot_widthvsn_single(df,alpha=0.10):
    method_names = [r'LR-$\chi^2$ test', r'Naive Z-test', r'Corrected Z-test', r'Parametric Bootstrap']
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

    type1error3=np.array(  # type I error at beta1=0.1 mu=0.25
        [
            [0.00133,	0.00133,	0.03833,	0.03],
            [0,	0.00467	,0.08333,	0.021],
            [0,	0.00367	,0.09667,	0.01333],
            [0,	0.00267,	0.10267,	0.00867],
            [0,0.00133,	0.10467,	0.00533],
            #[0,	0.00067,	0.10567	,0.002],
            #[0,	0.002,	0.113,	0.003]
        ]
    )

    type1error2=np.array(  # type I error at beta1=2.5 mu=0.25
        [
            [0.15867,	0.04267,	0.09033,	0.105],
            [0.20533,	0.06067,	0.094,	0.1],
            [0.26567,	0.06333,	0.09167,	0.09367],
            [0.38667,	0.07767,	0.09967,	0.09833],
            [0.56333,	0.092,	0.10333,	0.10333],
            #[0.699,	0.085,	0.09967,	0.09667],
            #[0.78767,	0.09033,	0.10133,	0.10333]
        ]
    )

    # Params, alpha=0.10
    #n_params = np.array([10, 25, 50, 100, 200, 300, 400])
    n_params = np.array([10, 25, 50, 100, 200])
    beta1_params = np.array([0.25, 2.5])
    alpha=0.1

    target_params = [[{
        "n": j,
        "beta": [i, 1.],
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
        plt.xlabel(r"Number of Bins $(n)$", fontsize=16)
        plot_cricvsn_single(criticalvalues[i],alpha)

        # width vs n
        #plt.subplot(2, 3, i * 3 + 3)
        #if i == 0:
        #    plt.title("Width")
        #plot_widthvsn_single(widths[i],alpha)


    plt.tight_layout()
    plt.savefig(f"figure/2x3vsn_to200.pdf", bbox_inches='tight')